r"""Beam search for hyperpixel layers"""


import datetime
import argparse
import os

from torch.utils.data import DataLoader
import torch
import time
from PIL import Image
from data import download
from model import scot_CAM, util, geometry
from model.objective import Objective
from common import loss
from common import utils
from common.logger import AverageMeter, Logger
from common.evaluation import Evaluator
import torch.optim as optim
from pprint import pprint
import wandb
from model.base.geometry import Geometry
from model import util

# DDP package
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import tempfile
from torch import distributed as dist
from collections import OrderedDict

wandb.login()

def all_reduce_results(results):
    """Coalesced mean all reduce over a dictionary of 0-dimensional tensors"""
    names, values = [], []
    for k, v in results.items():
        names.append(k)
        values.append(v)

    # Peform the actual coalesced all_reduce
    values = torch.stack([torch.tensor(v) for v in values], dim=0).cuda()
    dist.all_reduce(values, dist.ReduceOp.SUM)
    values.div_(dist.get_world_size())
    values = torch.chunk(values, values.size(0), dim=0)

    # Reconstruct the dictionary
    return OrderedDict((k, v.item()) for k, v in zip(names, values))

def train(epoch, model, dataloader, loss_func, optimizer, args, device):
    r"""Code for training SCOT"""
 
    model.module.train()
    dataloader.sampler.set_epoch(epoch)

    average_meter = AverageMeter(args.benchmark, dataloader.dataset.cls)
    total_steps = len(dataloader)

    losses = {}
    # Training
    for step, data in enumerate(dataloader):
        optimizer.zero_grad()
        iters = step + epoch * total_steps
        if args.use_wandb and dist.get_rank()==0:
            wandb.log({'iters':iters})

        data['src_img'] = data['src_img'].to(device)
        data['trg_img'] = data['trg_img'].to(device)
        data['src_kps'] = data['src_kps'].to(device)
        data['n_pts'] = data['n_pts'].to(device)
        data['trg_kps'] = data['trg_kps'].to(device)
        data['pckthres'] = data['pckthres'].to(device)

        if "src_mask" in data.keys():
            data["src_mask"] = data["src_mask"].to(device)
            data["trg_mask"] = data["trg_mask"].to(device)
        else:
            data["src_mask"] = None
            data["trg_mask"] = None

        # 1. forward pass
        src, trg = model(
            data['src_img'],
            data['trg_img'],
            args.classmap,
            data["src_mask"],
            data["trg_mask"],
            args.backbone,
            'train',
        )

        # {'box', 'feats', 'imsize', 'weights'}
        # 2. calculate sim
        assert args.loss_stage in ['sim', 'sim_geo', 'votes', 'votes_geo'], "Unrecognized loss stage"
        if 'sim' in args.loss_stage:
            cross_sim = model.module.calculate_sim(src['feats'], trg['feats'])
            if args.loss == 'weak':
                src_sim = model.module.calculate_sim(src['feats'], src['feats'])
                trg_sim = model.module.calculate_sim(trg['feats'], trg['feats'])

        if 'votes' in args.loss_stage:
            src_size = (src['feats'].size()[0], src['feats'].size()[1])
            trg_size = (trg['feats'].size()[0], trg['feats'].size()[1])
            cross_sim = model.module.calculate_votes(src['feats'], trg['feats'], args.epsilon, args.exp2, src_size, trg_size, src['weights'], trg['weights'])
            if args.loss == 'weak':
                src_sim = model.module.calculate_votes(src['feats'], src['feats'], args.epsilon, args.exp2, src_size, src_size, src['weights'], src['weights'])
                trg_sim = model.module.calculate_votes(trg['feats'], trg['feats'], args.epsilon, args.exp2, trg_size, trg_size, trg['weights'], trg['weights'])
        
        if 'geo' in args.loss_stage:
            cross_sim = model.module.calculate_votesGeo(cross_sim, src['imsize'], trg['imsize'], src['box'], trg['box'])
            if args.loss == 'weak':
                src_sim = model.module.calculate_votesGeo(src_sim, src['imsize'], src['imsize'], src['box'], src['box'])
                trg_sim = model.module.calculate_votesGeo(trg_sim, trg['imsize'], trg['imsize'], trg['box'], trg['box'])

        # 3. calculate loss
        src_center = 0
        trg_center = 0
        if args.loss == "strong_ce":
            src_center = geometry.center(src['box'])
            trg_center = geometry.center(trg['box'])
            data['src_kpidx'] = utils.match_idx(data['src_kps'], data['n_pts'], src_center)    
            data['trg_kpidx'] = utils.match_idx(data['trg_kps'], data['n_pts'], trg_center)
            # prediction and evaluation for current loss stage
            prd_kps = geometry.predict_kps(
                                src['box'],
                                trg['box'],
                                data["src_kps"],
                                data["n_pts"],
                                cross_sim.detach().clone(),
                                ) # predicted points
            eval_result = Evaluator.evaluate(prd_kps, data, args.loss) # return dict results
            loss = loss_func.compute_loss(cross_sim, eval_result, data['pckthres'], data['n_pts'])
            del prd_kps, eval_result
        elif args.loss == 'flow':
            data['flow'] = Geometry.KpsToFlow(data['src_kps'], data['trg_kps'], data['n_pts'])
            loss = loss_func.compute_loss(cross_sim, data['flow'], model.module.feat_size)
        elif args.loss == 'weak':
            disc_loss, match_loss = loss_func.compute_loss(cross_sim, src_sim, trg_sim, src['feats'], trg['feats'], args.weak_norm, args.temp)
            loss = args.weak_lambda * disc_loss + (1-args.weak_lambda) * match_loss
            losses['disc_loss'] = loss.item()
            losses['match_loss'] = match_loss.item()

        losses['total_loss'] = loss.item()

        # back propagation
        loss.backward()
        if args.use_grad_clip:
            torch.nn.utils.clip_grad.clip_grad_value_(model.module.parameters(), args.grad_clip)
        optimizer.step()
      

        # 4. collect results
        with torch.no_grad():
            prd_kps_list = {}
            eval_result_list = {}
            sim = model.module.calculate_sim(src['feats'], trg['feats'])
            src_size = (src['feats'].size()[0], src['feats'].size()[1])
            trg_size = (trg['feats'].size()[0], trg['feats'].size()[1])
            votes = model.module.calculate_votes(src['feats'], trg['feats'], args.epsilon, args.exp2, src_size, trg_size, src['weights'], trg['weights'])
            votes_geo = model.module.calculate_votesGeo(votes, src['imsize'], trg['imsize'], src['box'], trg['box'])
            for key, corr in zip(['sim', 'votes', 'votes_geo'], [sim, votes, votes_geo]):
                prd_kps = geometry.predict_kps(
                            src['box'],
                            trg['box'],
                            data["src_kps"],
                            data["n_pts"],
                            corr,
                            ) # predicted points
                eval_result = Evaluator.evaluate(prd_kps, data, args.loss) # return dict results
                # prd_kps_list[key] = prd_kps
                eval_result_list[key] = utils.mean(eval_result['pck'])
                del prd_kps

            # all reduce to update all processes
            eval_result_list = all_reduce_results(eval_result_list)
            losses = all_reduce_results(losses)
            average_meter.update(
                eval_result_list,
                data["pair_class"],
                losses['total_loss'],
            )
        
        # 5. print running pck, loss    
        if (step % 20 == 0) and dist.get_rank() == 0:
            average_meter.write_process(step, len(dataloader), epoch)

        # 6. draw weight map
        if (step % 40 == 0) and dist.get_rank() == 0:
            # 1. Draw weight map
            weight_map_path = os.path.join(Logger.logpath, "weight_map")
            os.makedirs(weight_map_path, exist_ok=True)
            weight_pth = utils.draw_weight_map(
                model.module.learner.layerweight.detach().clone().view(-1),
                epoch,
                step,
                weight_map_path,
            )

            if args.use_wandb:
                wandb.log(
                    {
                        "weight_map": wandb.Image(Image.open(weight_pth).convert("RGB")),
                    }
                )

        # 5. collect gradients
        if args.loss == 'weak':
            #print('disc_loss = %.2f, disc_grad = %.2f, match_loss = %.2f, match_grad = %.2f'%(disc_loss.item(), disc_loss.grad.item(), match_loss.item(), match_loss.grad.item()))
            if args.use_wandb:
                #wandb.log({'disc_grad':disc_loss.grad.item(), 'match_grad':match_loss.grad.item()})
                wandb.log({'disc_loss':disc_loss.item(), 'match_loss':match_loss.item()})
            del match_loss, disc_loss
                
        del src, trg, src_center, trg_center
        del loss, eval_result_list, prd_kps_list            
    
    # Draw class pck
    if False and (epoch % 2)==0:
        draw_class_pck_path = os.path.join(Logger.logpath, "draw_class_pck")
        os.makedirs(draw_class_pck_path, exist_ok=True)
        class_pth = utils.draw_class_pck(
            average_meter.sel_buffer["votes_geo"], draw_class_pck_path, epoch, step
        )
        if args.use_wandb:
            wandb.log(
                {
                    "class_pck": wandb.Image(Image.open(class_pth).convert("RGB")),
                }
            )

    # 3. log epoch loss, epoch pck
    if dist.get_rank() == 0:
        average_meter.write_result("Training", epoch)

    avg_loss = utils.mean(average_meter.loss_buffer)
    avg_pck = {'sim':0, 'votes':0, 'votes_geo':0}
    for key in ['sim', 'votes', 'votes_geo']:
        avg_pck[key] = utils.mean(average_meter.buffer[key])

    return avg_loss, avg_pck

def validate(epoch, model, dataloader, args, device):
    r"""Code for training SCOT"""
 
    model.module.eval()
    dataloader.sampler.set_epoch(epoch)

    average_meter = AverageMeter(args.benchmark, dataloader.dataset.cls)

    # Training
    for step, data in enumerate(dataloader):
        data['src_img'] = data['src_img'].to(device)
        data['trg_img'] = data['trg_img'].to(device)
        data['src_kps'] = data['src_kps'].to(device)
        data['n_pts'] = data['n_pts'].to(device)
        data['trg_kps'] = data['trg_kps'].to(device)
        data['pckthres'] = data['pckthres'].to(device)

        if "src_mask" in data.keys():
            data["src_mask"] = data["src_mask"].to(device)
            data["trg_mask"] = data["trg_mask"].to(device)
        else:
            data["src_mask"] = None
            data["trg_mask"] = None

        # 1. forward pass
        src, trg = model(
            data['src_img'],
            data['trg_img'],
            args.classmap,
            data["src_mask"],
            data["trg_mask"],
            args.backbone,
            'validate',
        )

        # {'box', 'feats', 'imsize', 'weights'}
        # 2. collect results
        if args.loss == "strong_ce":
            src_center = geometry.center(src['box'])
            trg_center = geometry.center(trg['box'])
            data['src_kpidx'] = utils.match_idx(data['src_kps'], data['n_pts'], src_center)    
            data['trg_kpidx'] = utils.match_idx(data['trg_kps'], data['n_pts'], trg_center)
        prd_kps_list = {}
        eval_result_list = {}
        sim = model.module.calculate_sim(src['feats'], trg['feats'])
        src_size = (src['feats'].size()[0], src['feats'].size()[1])
        trg_size = (trg['feats'].size()[0], trg['feats'].size()[1])
        votes = model.module.calculate_votes(src['feats'], trg['feats'], args.epsilon, args.exp2, src_size, trg_size, src['weights'], trg['weights'])
        votes_geo = model.module.calculate_votesGeo(votes, src['imsize'], trg['imsize'], src['box'], trg['box'])
        for key, corr in zip(['sim', 'votes', 'votes_geo'], [sim, votes, votes_geo]):
            prd_kps = geometry.predict_kps(
                        src['box'],
                        trg['box'],
                        data["src_kps"],
                        data["n_pts"],
                        corr,
                        ) # predicted points
            eval_result = Evaluator.evaluate(prd_kps, data, args.loss) # return dict results
            # prd_kps_list[key] = prd_kps
            eval_result_list[key] = utils.mean(eval_result['pck'])
            del prd_kps

        # all reduce to update all processes
        eval_result_list = all_reduce_results(eval_result_list)
        average_meter.update(
            eval_result_list,
            data["pair_class"],
        )
        
        # 3. print running pck, loss    
        if (step % 1 == 0) and dist.get_rank() == 0:
            average_meter.write_process(step, len(dataloader), epoch)

                
        del src, trg
        del eval_result_list, prd_kps_list            
    
    # log epoch loss, epoch pck
    if dist.get_rank() == 0:
        average_meter.write_result("Validation", epoch)

    avg_loss = utils.mean(average_meter.loss_buffer)
    avg_pck = {'sim':0, 'votes':0, 'votes_geo':0}
    for key in ['sim', 'votes', 'votes_geo']:
        avg_pck[key] = utils.mean(average_meter.buffer[key])

    return avg_loss, avg_pck

def init_distributed_mode(args):
    """ init for distribute mode """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        print('local_rank = ', int(os.environ['LOCAL_RANK']), ' rank =', args.rank)
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return
    
    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    dist.barrier()

def make_dataloader(args, rank, world_size):
    num_workers = 16 if torch.cuda.is_available() else 8
    pin_memory = True if torch.cuda.is_available() else False
    
    Logger.info("loading %s dataset"%(args.benchmark))
    trn_ds = download.load_dataset(args.benchmark, args.datapath, args.thres, args.split, args.cam, img_side=img_side, use_resize=True, use_batch=args.use_batch)
    trn_sampler = DistributedSampler(trn_ds)
    trn_dl = DataLoader(dataset=trn_ds, batch_size=args.batch_size, num_workers=num_workers, sampler=trn_sampler, pin_memory=pin_memory)
    
    if args.split not in ["val", "trnval"]:
        val_ds = download.load_dataset(args.benchmark, args.datapath, args.thres, "val", args.cam, img_side=img_side, use_resize=True, use_batch=args.use_batch)
        val_sampler = DistributedSampler(val_ds)
        val_dl = DataLoader(dataset=val_ds, batch_size=args.batch_size, num_workers=num_workers, sampler=val_sampler, pin_memory=pin_memory)
    
    Logger.info("loading dataset finished")

    return trn_dl, val_dl

def main(args):
    ####    1. init DDP and Logger  ####
    Logger.initialize(args, training=True)

    if args.distributed:
        Logger.info('[INFO] turn on distributed train')
    else:
        Logger.info('[INFO] turn off distributed train')

    rank = dist.get_rank()
    local_rank = args.gpu
    device = torch.device("cuda:{}".format(local_rank))
    world_size = dist.get_world_size()
    util.fix_randseed(seed=(0+rank))

    n_layers = {"resnet50": 17, "resnet101": 34, "fcn101": 34}
    hyperpixels = list(range(n_layers[args.backbone]))


    ####    2. make dataloader  ####
    trn_dl, val_dl = make_dataloader(args, rank, world_size)


    ####    3. Model intialization ####
    model = scot_CAM.SCOT_CAM(
        args.backbone,
        hyperpixels,
        args.benchmark,
        args.cam,
        args.use_xavier,
        args.weight_thres,
        args.select_all
    ).to(device)

    # resume the model
    if args.resume and rank == 0:
        Logger.info('Loading snapshot from %s'%(args.resume_path))
        model.module.load_state_dict(torch.load(args.resume_path))

    # load backbone 
    if args.selfsup in ['dino', 'denseCL'] and rank == 0:
        Logger.info('Loading backbone from %s'%(args.backbone_path))
        pretrained_backbone = torch.load(args.backbone_path)
        backbone_keys = list(model.module.backbone.state_dict().keys())
            
        if 'state_dict' in pretrained_backbone:
            model.module.load_backbone(pretrained_backbone['state_dict'])
            load_keys = list(pretrained_backbone['state_dict'].keys())
        else:
            model.module.load_backbone(pretrained_backbone)
            load_keys = list(pretrained_backbone.keys())
        missing_keys = [i for i in backbone_keys if i not in load_keys]
        print(missing_keys)

    if args.distributed:
        model = DDP(model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True)

    ####    4. loss function, optimizer and scheduler ####    
    Objective.initialize(target_rate=0.5, alpha=args.alpha)
    if args.loss == "weak":
        loss_func = loss.WeakLoss()
    elif args.loss == "strong_ce":
        loss_func = loss.StrongCELoss()
    elif args.loss == "flow":
        loss_func = loss.StrongFlowLoss()
    else:
        raise ValueError("Unrecognized objective loss")

    assert args.optimizer in ["sgd", "adam"], "Unrecognized model type"
    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.module.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = optim.AdamW(model.module.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.use_scheduler:
        assert args.scheduler in ["cycle", "step", "cosin"], "Unrecognized model type" 
        if args.scheduler == "cycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, epochs=args.epochs, steps_per_epoch=len(trn_dl))
        elif args.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.step_gamma)

    ####    7. Evaluator #### 
    Evaluator.initialize(args.alpha)

    ####    8. wandb parameters ####
    if args.use_wandb and rank == 0:
        wandb_name = "%.e_%s_%s_%s"%(args.lr, args.loss_stage, args.loss, args.optimizer)
        if args.use_scheduler:
            wandb_name += "_%s"%(args.scheduler)
        if args.optimizer == "sgd":
            wandb_name = wandb_name + "_m%.2f"%(args.momentum)
        wandb_name = wandb_name + "_bsz%d"%(args.batch_size)
        
        if args.use_scot2:
            wandb_name = wandb_name + "_scot2"
        wandb_name = wandb_name + "_%s_%s_%s_alp%.2f"%(args.selfsup, args.backbone, args.split, args.alpha)

        run = wandb.init(project=args.wandb_proj, config=args, id=args.run_id, resume="allow", name=wandb_name)

        wandb.define_metric("iters")
        wandb.define_metric("running_trn_avg_loss", step_metric="iters")
        wandb.define_metric("running_avg_pck_sim", step_metric="iters")
        wandb.define_metric("running_avg_pck_votes", step_metric="iters")
        wandb.define_metric("running_avg_pck_votes_geo", step_metric="iters")
        
        wandb.define_metric("epochs")
        wandb.define_metric("trn_loss", step_metric="epochs")
        wandb.define_metric("trn_pck_sim", step_metric="epochs")
        wandb.define_metric("trn_pck_votes", step_metric="epochs")
        wandb.define_metric("trn_pck_votes_geo", step_metric="epochs")
        
        wandb.define_metric("val_loss", step_metric="epochs")
        wandb.define_metric("val_pck_sim", step_metric="epochs")
        wandb.define_metric("val_pck_votes", step_metric="epochs")
        wandb.define_metric("val_pck_votes_geo", step_metric="epochs")

        if args.loss == 'weak':
            wandb.define_metric("disc_grad", step_metric="iters")
            wandb.define_metric("match_grad", step_metric="iters")
            wandb.define_metric("disc_loss", step_metric="iters")
            wandb.define_metric("match_loss", step_metric="iters")
        
    ####    9. Train SCOT  ####
    best_val_pck = float("-inf")
    log_benchmark = {}
     
    Logger.info("Training Start")
    train_started = time.time()
            
    for epoch in range(args.start_epoch, args.epochs):

        # training
        trn_loss, trn_pck = train(epoch, model, trn_dl, loss_func, optimizer, args, device)
        log_benchmark["trn_loss"] = trn_loss
        log_benchmark["trn_pck_sim"] = trn_pck['sim']
        log_benchmark["trn_pck_votes"] = trn_pck['votes']
        log_benchmark["trn_pck_votes_geo"] = trn_pck['votes_geo']
        
        # validation
        if args.split not in ["val", "trnval"]:
            with torch.no_grad():
                val_loss, val_pck = validate(epoch, model, val_dl, args, device)
                log_benchmark["val_loss"] = val_loss
                log_benchmark["val_pck_sim"] = val_pck['sim']
                log_benchmark["val_pck_votes"] = val_pck['votes']
                log_benchmark["val_pck_votes_geo"] = val_pck['votes_geo']
                
        if args.use_scheduler:
                # lrs.append(utils.get_lr(optimizer))
                scheduler.step() # update lr batch-by-batch
                log_benchmark["lr"] = scheduler.get_last_lr()[0]

        # save the best model
        if args.split in ['old_trn', 'trn']:
            model_pck = val_pck
        else:
            model_pck = trn_pck
            
        # save_e = 1 if args.split == "trnval" else 5
        # if (epoch%save_e)==0:
        #     Logger.save_epoch(model, epoch, model_pck["votes_geo"])
            
        if model_pck["votes_geo"] > best_val_pck and rank == 0:
            Logger.save_model(model.module, epoch, model_pck["votes_geo"], best_val_pck)
            best_val_pck = model_pck["votes_geo"]

        if args.use_wandb and rank == 0:
            wandb.log({'epochs':epoch})
            wandb.log(log_benchmark)

        if rank == 0:
            time_message = 'Training %d epochs took:%4.3f\n' % (epoch+1, (time.time()-train_started)/60) + ' minutes'
            Logger.info(time_message)
    

    Logger.info("==================== Finished training ====================")
    
if __name__ == "__main__":

    # Arguments parsing
    # fmt: off
    parser = argparse.ArgumentParser(description="SCOT Training Script")

    parser.add_argument('--gpu', type=str, default='0', help='GPU id')
    parser.add_argument('--datapath', type=str, default='./Datasets_SCOT') 
    parser.add_argument('--benchmark', type=str, default='pfpascal')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--selfsup', type=str, default='sup', choices=['sup', 'dino', 'denseCL'], help='supervised or self-supervised backbone')
    parser.add_argument('--thres', type=str, default='auto', choices=['auto', 'img', 'bbox'])
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--split', type=str, default='trn', help='trn, val, test, old_trn, trn_val') 

    # Training parameters
    parser.add_argument('--loss', type=str, default='strong_ce', choices=['weak', 'strong_ce', 'flow'])
    parser.add_argument('--lr', type=float, default=0.01) 
    parser.add_argument('--lr_backbone', type=float, default=0.0) 
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--optimizer', type=str, default="sgd", choices=["sgd", "adam"])
    parser.add_argument('--weight_decay', type=float, default=0.00, help='weight decay (default: 0.00)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (default: 0.9)')
    parser.add_argument("--use_scheduler", type=util.boolean_string, nargs="?", default=False)
    parser.add_argument("--scheduler", type=str, default="cycle", choices=["step", "cycle", "cosine"])
    parser.add_argument('--step_size', type=int, default=16, help='hyperparameters for step scheduler')
    parser.add_argument('--step_gamma', type=float, default=0.1, help='hyperparameters for step scheduler')
    
    parser.add_argument("--use_grad_clip", type=util.boolean_string, nargs="?", default=False)
    parser.add_argument("--grad_clip", type=float, default=0.1, help='gradient clip threshold') 
    parser.add_argument("--use_wandb", type= utils.boolean_string, nargs="?", default=False)
    parser.add_argument("--use_xavier", type= utils.boolean_string, nargs="?", default=False)
    parser.add_argument('--loss_stage', type=str, default="sim", choices=['sim', 'sim_geo', 'votes', 'votes_geo'])
    parser.add_argument("--resume", type= utils.boolean_string, nargs="?", const=True, default=False)
    parser.add_argument('--resume_path', type=str, default='')
    parser.add_argument('--backbone_path', type=str, default='./backbone')
    parser.add_argument('--weight_thres', type=float, default=0.00, help='weight_thres (default: 0.00)')
    parser.add_argument('--select_all', type=float, default=1.01, help='selec all probability (default: 1.0)')
    
    parser.add_argument('--img_side', type=str, default='(300)')
    parser.add_argument("--use_batch", type= utils.boolean_string, nargs="?", default=False)
    parser.add_argument("--trg_cen", type= utils.boolean_string, nargs="?", default=False)
    parser.add_argument("--use_scot2", type= utils.boolean_string, nargs="?", default=False)

    # Algorithm parameters
    parser.add_argument('--sim', type=str, default='OTGeo', help='Similarity type: OT, OTGeo, cos, cosGeo')
    parser.add_argument('--exp1', type=float, default=1.0, help='exponential factor on initial cosine cost')
    parser.add_argument('--exp2', type=float, default=1.0, help='exponential factor on final OT scores')
    parser.add_argument('--epsilon', type=float, default=0.05, help='epsilon for Sinkhorn Regularization')
    parser.add_argument('--classmap', type=int, default=0, help='class activation map: 0 for none, 1 for using CAM')
    parser.add_argument('--cam', type=str, default='', help='activation map folder, empty for end2end computation')
    # default is the value that the attribute gets when the argument is absent. const is the value it gets when given.

    parser.add_argument('--run_id', type=str, default='', help='run_id')
    parser.add_argument('--wandb_proj', type=str, default='', help='wandb project name')
    parser.add_argument("--weak_norm", type=str, default="l1", choices=["l1", "linear", "softmax"])
    parser.add_argument('--weak_lambda', type=float, default=0.5, help='lambda for weak loss (default: 0.5)')
    parser.add_argument('--temp', type=float, default=0.05, help='lambda for weak loss (default: 0.5)')

    # Arguments for distributed data parallel
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int, help='numer of distributed processes')
    parser.add_argument("--local_rank", required=True, type=int, help='local rank for DistributedDataParallel')

    args = parser.parse_args()
    
    init_distributed_mode(args)


    if args.use_wandb and args.run_id == '':
        args.run_id = wandb.util.generate_id()
    img_side = util.parse_string(args.img_side)
    if isinstance(img_side, int):
        # use target rhf center if only scale the max_side
        args.trg_cen = True
        args.use_batch = False
    else:
        args.trg_cen = False
        args.use_batch = True

    main(args)