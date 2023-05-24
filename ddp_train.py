import datetime
import argparse
import os
import sys
from torch.utils.data import DataLoader
import torch
import time
from PIL import Image
from data.download import load_dataset
from model import scot_CAM, geometry
from common.loss import StrongCrossEntropyLoss, StrongFlowLoss, WeakDiscMatchLoss
from common.utils import match_idx, mean, draw_weight_map, boolean_string, fix_randseed, parse_string, NewAverageMeter, ProgressMeter, reduce_tensor, Summary
from common.logger import Logger
from common.evaluation import Evaluator
import torch.optim as optim
from pprint import pprint
import wandb
import re
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset
import numpy as np

# DDP package
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as dist

wandb.login()


def train(args, model, criterion, dataloader, optimizer, epoch):
    # Logger.info(f'Current learning rate for different parameter groups: {[it["lr"] for it in optimizer.param_groups]}')
    # average_meter = AverageMeter(args.benchmark, dataloader.dataset.cls)

    # 1. Meter
    loss_meter = NewAverageMeter('loss', ':4.2f')
    pck_meter = NewAverageMeter('pck', ':4.2f')
    progress_list = [loss_meter, pck_meter]
    if args.criterion == "weak":
        discSelf_meter = NewAverageMeter('Selfloss', ':4.2f')
        discCross_meter = NewAverageMeter('Crossloss', ':4.2f') 
        match_meter = NewAverageMeter('matchloss', ':4.2f') 
        progress_list += [discSelf_meter, discCross_meter, match_meter]
        
        if args.collect_grad:
            discSelfG_meter = NewAverageMeter('SelfG', ':4.2f', summary_type=Summary.VAL)
            discCrossG_meter = NewAverageMeter('CrossG', ':4.2f', summary_type=Summary.VAL) 
            matchG_meter = NewAverageMeter('matchG', ':4.2f', summary_type=Summary.VAL) 
            progress_list += [discSelfG_meter, discCrossG_meter, matchG_meter]

    progress = ProgressMeter(len(dataloader), progress_list, prefix="Epoch[{}]".format(epoch))

    # 2. model status
    model.module.backbone.eval()
    model.module.learner.train()
    optimizer.zero_grad()


    total_steps = len(dataloader)
    Pointtime = {'Point-1':0, 'Point-2':0,'Point-3':0,'Point-4':0,'Point-5':0,'Point-6':0,}
    for step, data in enumerate(dataloader):

        iters = step + epoch * total_steps
        if args.use_wandb and dist.get_rank() == 0:
            wandb.log({"iters": iters})

        bsz = data["src_img"].size(0)
        if args.use_negative:
            shifted_idx = np.roll(np.arange(bsz), -1)
            trg_img_neg = data['trg_img'][shifted_idx].clone()
            trg_cls_neg = data['category_id'][shifted_idx].clone()
            neg_subidx = (data['category_id'] - trg_cls_neg) != 0

            src_img = torch.cat([data['src_img'], data['src_img'][neg_subidx]], dim=0).cuda(non_blocking=True)
            trg_img = torch.cat([data['trg_img'], trg_img_neg[neg_subidx]], dim=0).cuda(non_blocking=True)

            del trg_img_neg

            if "src_mask" in data.keys():
                trg_mask_neg = data['trg_mask'][shifted_idx].clone()
                src_mask = torch.cat([data['src_mask'], data['src_mask'][neg_subidx]], dim=0).cuda(non_blocking=True)
                trg_mask = torch.cat([data['trg_mask'], trg_mask_neg[neg_subidx]], dim=0).cuda(non_blocking=True)

                del trg_mask_neg
            else:
                src_mask, trg_mask = None, None
            
            num_negatives = neg_subidx.sum()
        
        else:
            num_negatives = 0
            src_img = data["src_img"].cuda(non_blocking=True)
            trg_img = data["trg_img"].cuda(non_blocking=True)

            if "src_mask" in data.keys():
                src_mask = data["src_mask"].cuda(non_blocking=True)
                trg_mask = data["trg_mask"].cuda(non_blocking=True)
            else:
                src_mask, trg_mask = None, None
        
        data["src_kps"] = data["src_kps"].cuda(non_blocking=True)
        data["n_pts"] = data["n_pts"].cuda(non_blocking=True)
        data["trg_kps"] = data["trg_kps"].cuda(non_blocking=True)
        data["pckthres"] = data["pckthres"].cuda(non_blocking=True)       

        # 1. compute output
        start_time = time.time()
        src, trg = model(
            src_img,
            trg_img,
            args.classmap,
            src_mask,
            trg_mask,
            args.backbone,
            'train',
        )
        Pointtime['Point-2'] += ((time.time()-start_time)/60)

        # {'box', 'feats', 'imsize', 'weights'}
        # 2. calculate sim
        start_time = time.time()
        assert args.loss_stage in ["sim", "sim_geo", "votes", "votes_geo",], "Unknown loss stage"
        
        weak_mode = True if args.criterion == "weak" and (args.weak_lambda[0]>0.0) else False
        
        if "sim" in args.loss_stage:
            cross_sim, src_sim, trg_sim = model.module.calculate_sim(src["feats"], trg["feats"], weak_mode, bsz)

        if "votes" in args.loss_stage:
            src_size = (src["feats"].size()[0], src["feats"].size()[1])
            trg_size = (trg["feats"].size()[0], trg["feats"].size()[1])
            cross_sim, src_sim, trg_sim = model.module.calculate_votes(
                src["feats"], trg["feats"], args.epsilon, args.exp2, src_size, trg_size, src["weights"], trg["weights"], weak_mode, bsz)

        if "geo" in args.loss_stage:
            cross_sim, src_sim, trg_sim = model.module.calculate_votesGeo(
                cross_sim, src_sim, trg_sim, src["imsize"], trg["imsize"], src["box"], trg["box"]
            )
        Pointtime['Point-3'] += ((time.time()-start_time)/60)

        start_time = time.time()
        # 3. calculate loss
        src_center = 0
        trg_center = 0
        if args.criterion == "strong_ce":
            with torch.no_grad():
                src_center = geometry.center(src["box"])
                trg_center = geometry.center(trg["box"])
                data["src_kpidx"] = match_idx(data["src_kps"], data["n_pts"], src_center)
                data["trg_kpidx"] = match_idx(data["trg_kps"], data["n_pts"], trg_center)
                # prediction and evaluation for current loss stage
                prd_kps = geometry.predict_kps(
                    src["box"], trg["box"], data["src_kps"],
                    data["n_pts"], cross_sim.detach().clone(),
                )  # predicted points
                eval_result = Evaluator.eval_kps_transfer(
                    prd_kps, data, args.criterion
                )  # return dict results
            loss = criterion(
                cross_sim, eval_result['easy_match'], eval_result['hard_match'], data["pckthres"], data["n_pts"]
            )
            del prd_kps, eval_result, src_center, trg_center
        elif args.criterion == "flow":
            pass
        elif args.criterion == "weak":
            task_loss = criterion(
                cross_sim, src_sim, trg_sim,
                src["feats"], trg["feats"], bsz, num_negatives
            )

            discSelf_meter.update(task_loss[0].item(), bsz)
            discCross_meter.update(task_loss[1].item(), bsz)
            match_meter.update(task_loss[2].item(), bsz)

            loss = (args.weak_lambda * task_loss).sum()

            del src_sim, trg_sim
        del cross_sim

        loss_meter.update(loss.item(), bsz)
        
        if args.criterion == "weak" and (step%10 == 0) and args.collect_grad:
            GW_t = []
            for i in range(3):
                # get the gradient of this task loss with respect to the shared parameters
                GiW_t = torch.autograd.grad(
                    task_loss[i], model.module.learner.parameters(), retain_graph=True)[0]
                                
                # GiW_t is tuple
                # compute the norm
                dist.barrier()     
                dist.all_reduce(GiW_t, op=dist.ReduceOp.SUM)
                GiW_t /= dist.get_world_size()
                GW_t.append(torch.norm(GiW_t).item())
                # print('gwt', GiW_t)

                # task_loss[i].backward(retain_graph=True)
                # print('grad', model.module.learner.layerweight.grad)
                # model.module.learner.layerweight.grad.zero_()
            del GiW_t
            discSelfG_meter.update(GW_t[0], bsz)
            discCrossG_meter.update(GW_t[1], bsz)
            matchG_meter.update(GW_t[2], bsz)
            if args.use_wandb and dist.get_rank() == 0:
                wandb.log({"discSelfGrad": GW_t[0], "discCrossGrad": GW_t[1], "matchGrad": GW_t[2]})

        # back propagation
        optimizer.zero_grad()   
        loss.backward()
        if args.use_grad_clip:
            torch.nn.utils.clip_grad.clip_grad_value_(model.parameters(), args.grad_clip)
        optimizer.step()
        del loss

        # 4. collect results
        with torch.no_grad():
            batch_pck = 0
            src_size = (src["feats"].size()[0], src["feats"].size()[1])
            trg_size = (trg["feats"].size()[0], trg["feats"].size()[1])
            votes, _, _ = model.module.calculate_votes(
                src["feats"], trg["feats"], args.epsilon, args.exp2, src_size, trg_size, src["weights"], trg["weights"], False, bsz)
            votes_geo, _, _ = model.module.calculate_votesGeo(
                votes, None, None, src["imsize"], trg["imsize"], src["box"], trg["box"])
            prd_kps = geometry.predict_kps(
                src["box"], trg["box"], data["src_kps"],
                data["n_pts"], votes_geo,
            )  # predicted points
            eval_result = Evaluator.eval_kps_transfer(
                prd_kps, data, args.criterion, pck_only=True
            )  # return dict results
            # prd_kps_list[key] = prd_kps
            batch_pck = mean(eval_result["pck"])
            del votes, votes_geo, eval_result, prd_kps

            pck_meter.update(batch_pck, bsz)
 
            del batch_pck

        # 5. print running pck, loss
        if (step % 100 == 0) and dist.get_rank() == 0:
            progress.display(step+1)

        del src, trg, data
    # torch.cuda.empty_cache()

    # Draw class pck
    # if False and (epoch % 2)==0:
    #     draw_class_pck_path = os.path.join(Logger.logpath, "draw_class_pck")
    #     os.makedirs(draw_class_pck_path, exist_ok=True)
    #     class_pth = utils.draw_class_pck(
    #         average_meter.sel_buffer["votes_geo"], draw_class_pck_path, epoch, step
    #     )
    #     if args.use_wandb and dist.get_rank() == 0:
    #         wandb.log(
    #             {
    #                 "class_pck": wandb.Image(Image.open(class_pth).convert("RGB")),
    #             }
    #         )

    # 6. draw weight map
    if dist.get_rank() == 0:
        # 1. Draw weight map
        weight_map_path = os.path.join(Logger.logpath, "weight_map")
        os.makedirs(weight_map_path, exist_ok=True)
        weight_pth = draw_weight_map(
            model.module.learner.layerweight.detach().clone(),
            epoch, weight_map_path)
        if args.use_wandb:
            wandb.log({"weight_map": wandb.Image(Image.open(weight_pth).convert("RGB"))})

    dist.barrier()
    # 3. log epoch loss, epoch pck
    loss_meter.all_reduce()
    pck_meter.all_reduce() 
    # 7. collect gradients
    if args.criterion == "weak":
        discSelf_meter.all_reduce()
        discCross_meter.all_reduce()
        match_meter.all_reduce()
        if args.use_wandb and dist.get_rank() == 0:
            wandb.log({"discSelf_loss": discSelf_meter.avg, "discCross_loss": discCross_meter.avg, "match_loss": match_meter.avg})
    
    return loss_meter.avg, pck_meter.avg

def validate(args, model, criterion, dataloader, epoch, aux_val_loader=None):
    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            for step, data in enumerate(loader):
                step = base_progress + step
                data["src_img"] = data["src_img"].cuda(non_blocking=True)
                data["trg_img"] = data["trg_img"].cuda(non_blocking=True)
                data["src_kps"] = data["src_kps"].cuda(non_blocking=True)
                data["n_pts"] = data["n_pts"].cuda(non_blocking=True)
                data["trg_kps"] = data["trg_kps"].cuda(non_blocking=True)
                data["pckthres"] = data["pckthres"].cuda(non_blocking=True)
                if "src_mask" in data.keys():
                    data["src_mask"] = data["src_mask"].cuda(non_blocking=True)
                    data["trg_mask"] = data["trg_mask"].cuda(non_blocking=True)
                else:
                    data["src_mask"], data["trg_mask"] = None, None
                bsz = data["src_img"].size(0)

                # 1. forward pass
                src, trg = model(
                    data["src_img"],
                    data["trg_img"],
                    args.classmap,
                    data["src_mask"],
                    data["trg_mask"],
                    args.backbone,
                    'eval',
                )

                # {'box', 'feats', 'imsize', 'weights'}
                # 2. calculate sim
                assert args.loss_stage in ["sim", "sim_geo", "votes", "votes_geo",], "Unknown loss stage"
                cross_sim, src_sim, trg_sim = 0, 0, 0
                weak_mode = True if args.criterion == "weak" and (args.weak_lambda[0]>0.0) else False
                if "sim" in args.loss_stage:
                    cross_sim, src_sim, trg_sim = model.module.calculate_sim(src["feats"], trg["feats"], weak_mode, bsz)

                if "votes" in args.loss_stage:
                    src_size = (src["feats"].size()[0], src["feats"].size()[1])
                    trg_size = (trg["feats"].size()[0], trg["feats"].size()[1])
                    cross_sim, src_sim, trg_sim = model.module.calculate_votes(
                        src["feats"], trg["feats"], args.epsilon, args.exp2, src_size, trg_size, src["weights"], trg["weights"], weak_mode, bsz)

                if "geo" in args.loss_stage:
                    cross_sim, src_sim, trg_sim = model.module.calculate_votesGeo(
                        cross_sim, src_sim, trg_sim, src["imsize"], trg["imsize"], src["box"], trg["box"]
                    )

                # 3. calculate loss
                src_center = 0
                trg_center = 0
                if args.criterion == "strong_ce":
                    
                    src_center = geometry.center(src["box"])
                    trg_center = geometry.center(trg["box"])
                    data["src_kpidx"] = match_idx(data["src_kps"], data["n_pts"], src_center)
                    data["trg_kpidx"] = match_idx(data["trg_kps"], data["n_pts"], trg_center)
                    # prediction and evaluation for current loss stage
                    prd_kps = geometry.predict_kps(
                        src["box"], trg["box"], data["src_kps"],
                        data["n_pts"], cross_sim.detach().clone(),
                    )  # predicted points
                    eval_result = Evaluator.eval_kps_transfer(
                        prd_kps, data, args.criterion
                    )  # return dict results
                    loss = criterion(
                        cross_sim, eval_result['easy_match'], eval_result['hard_match'], data["pckthres"], data["n_pts"]
                    )
                    del prd_kps, eval_result, src_center, trg_center
                elif args.criterion == "flow":
                    pass
                elif args.criterion == "weak":
                    task_loss = criterion(
                        cross_sim, src_sim, trg_sim,
                        src["feats"], trg["feats"], bsz, num_negatives=0
                    )

                    loss = (args.weak_lambda * task_loss).sum()
                
                    del src_sim, trg_sim
                del cross_sim

                loss_meter.update(loss.item(), bsz)
                del loss

                # 4. collect results
                batch_pck = 0
                src_size = (src["feats"].size()[0], src["feats"].size()[1])
                trg_size = (trg["feats"].size()[0], trg["feats"].size()[1])
                votes, _, _ = model.module.calculate_votes(
                    src["feats"], trg["feats"], args.epsilon, args.exp2, src_size, trg_size, src["weights"], trg["weights"], False, bsz)
                votes_geo, _, _ = model.module.calculate_votesGeo(
                    votes, None, None, src["imsize"], trg["imsize"], src["box"], trg["box"])
                prd_kps = geometry.predict_kps(
                    src["box"], trg["box"], data["src_kps"],
                    data["n_pts"], votes_geo,
                )  # predicted points
                eval_result = Evaluator.eval_kps_transfer(
                    prd_kps, data, args.criterion, pck_only=True
                )  # return dict results
                # prd_kps_list[key] = prd_kps
                batch_pck = mean(eval_result["pck"])
                del votes, votes_geo, eval_result, prd_kps

                pck_meter.update(batch_pck, bsz)
   
                del batch_pck, data, src, trg

    loss_meter = NewAverageMeter('loss', ':4.2f')
    pck_meter = NewAverageMeter('pck', ':4.2f')
    progress_list = [loss_meter, pck_meter]
    progress = ProgressMeter(
        len(dataloader) + (args.distributed and (len(dataloader.sampler) * args.world_size < len(dataloader.dataset))), 
        progress_list, prefix="Test[{}]: ".format(epoch))

    # switch to evaluate mode
    model.module.backbone.eval()
    model.module.learner.eval()

    run_validate(dataloader)

    dist.barrier()
    loss_meter.all_reduce()
    pck_meter.all_reduce()

    if aux_val_loader is not None:
        run_validate(aux_val_loader, len(dataloader))

    dist.barrier()
    progress.display_summary()

    avg_loss = loss_meter.avg
    avg_pck = pck_meter.avg

    return avg_loss, avg_pck

def init_distributed_mode(args):
    """init for distribute mode"""
    # launched with torch.distributed.launch
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
    # launched with submitit on a slurm cluster
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.local_rank = args.rank % torch.cuda.device_count()
    elif torch.cuda.is_available():
        print("Will run the code on one GPU.")
        args.rank, args.local_rank, args.world_size = 0, 0, 1
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
    else:
        print("Does not support training without GPU.")
        sys.exit(1)

    args.dist_backend = "nccl"
    args.distributed = True
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.local_rank)
    print(
        "| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True
    )
    dist.barrier()

def build_dataloader(args, rank, world_size):
    num_workers = 16 if torch.cuda.is_available() else 8
    pin_memory = True if torch.cuda.is_available() else False

    Logger.info("Loading %s dataset" % (args.benchmark))

    # training set
    train_dataset = load_dataset(
        args.benchmark,
        args.datapath,
        args.thres,
        "trn",
        args.cam,
        output_image_size=args.output_image_size,
        use_resize=True,
        use_batch=args.use_batch,
    )
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=pin_memory, sampler=train_sampler)

    # validation set
    val_dataset = load_dataset(
        args.benchmark,
        args.datapath,
        args.thres,
        "val",
        args.cam,
        output_image_size=args.output_image_size,
        use_resize=True,
        use_batch=args.use_batch,
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True
    )
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, sampler=val_sampler,
    )

    Logger.info(
        f"Data loaded: there are {len(train_loader.dataset)} train images and {len(val_loader.dataset)} val images."
    )
    if len(val_loader.sampler) * world_size < len(val_loader.dataset):
        aux_val_dataset = Subset(val_loader.dataset, range(len(val_loader.sampler)*world_size, len(val_loader.dataset)))
        aux_val_loader = DataLoader(aux_val_dataset, batch_size=args.batch_size, shuffle=False,num_workers=num_workers, pin_memory=pin_memory)

        Logger.info('Create Subset: ', len(val_loader.sampler),  world_size, len(val_loader.dataset))
    else:
        aux_val_loader = None
    return train_loader, val_loader, aux_val_loader

def build_scheduler(args, optimizer, n_iter_per_epoch, config=None):
    # modified later
    # num_steps = int(config.TRAIN.EPOCHS * n_iter_per_epoch)
    # warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)
    # decay_steps = int(config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS * n_iter_per_epoch)
    # multi_steps = [i * n_iter_per_epoch for i in config.TRAIN.LR_SCHEDULER.MULTISTEPS]

    lr_scheduler = None
    if args.scheduler == "cycle":
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, args.lr, epochs=args.epochs, steps_per_epoch=n_iter_per_epoch
        )
    elif args.scheduler == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.step_gamma
        )

    return lr_scheduler

def build_optimizer(args, model):
    """
    Build optimizer, e.g., sgd, adamw.
    """
    assert args.optimizer in ["sgd", "adamw"], "Unknown optimizer type"
    optimizer = None

    # take parameters
    parameter_group_names = {"params": []}
    parameter_group_vars = {"params": []}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        parameter_group_names["params"].append(name)
        parameter_group_vars["params"].append(param)

    # make optimizers
    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            [parameter_group_vars], lr=args.lr, momentum=args.momentum
        )
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(
            [parameter_group_vars], lr=args.lr, weight_decay=args.weight_decay
        )

    return optimizer

def load_checkpoint(args, model, optimizer, lr_scheduler):
    max_pck = 0.0
    if args.resume:
        Logger.info(f">>>>>>>>>> Resuming from {args.resume} ..........")
        checkpoint = torch.load(args.resume, map_location="cpu")

        msg = model.load_state_dict(checkpoint["model"], strict=False)
        Logger.info(msg)
        
        if (
            not args.eval_mode
            and "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1

            Logger.info(
                f"=> loaded successfully '{args.resume}' (epoch {checkpoint['epoch']})"
            )

            if "max_pck" in checkpoint:
                max_pck = checkpoint["max_pck"]
            else:
                max_pck = 0.0

        del checkpoint
        torch.cuda.empty_cache()

    # load backbone
    if args.selfsup in ["dino", "denseCL"]:
        Logger.info("Loading backbone from %s" % (args.backbone_path))
        pretrained_backbone = torch.load(args.backbone_path, map_location="cpu")
        backbone_keys = list(model.backbone.state_dict().keys())

        if "state_dict" in pretrained_backbone:
            model.load_backbone(pretrained_backbone["state_dict"], strict=False)
            load_keys = list(pretrained_backbone["state_dict"].keys())
        else:
            model.load_backbone(pretrained_backbone)
            load_keys = list(pretrained_backbone.keys())
        missing_keys = [i for i in backbone_keys if i not in load_keys]
        Logger.info("missing keys in loaded backbone: %s" % (missing_keys))

        del pretrained_backbone
        torch.cuda.empty_cache()

    return max_pck

def build_wandb(args, rank):
    if args.use_wandb and rank == 0:
        wandb_name = "%.e_%s_%s_%s" % (
            args.lr,
            args.loss_stage,
            args.criterion,
            args.optimizer,
        )
        if args.scheduler != "none":
            wandb_name += "_%s" % (args.scheduler)
        if args.optimizer == "sgd":
            wandb_name = wandb_name + "_m%.2f" % (args.momentum)
        
        if args.use_negative:
            wandb_name += "_bsz%d-neg" % (args.batch_size*2)
            args.batch_size = args.batch_size*2
        else:
            wandb_name += "_bsz%d" % (args.batch_size)

        if args.criterion == 'weak':
            wandb_name += ("_%s_tp%.2f"%(args.weak_lambda, args.temp))

        _wandb = wandb.init(
            project=args.wandb_proj,
            config=args,
            id=args.run_id,
            resume="allow",
            name=wandb_name,
        )

        wandb.define_metric("iters")
        wandb.define_metric("running_avg_loss", step_metric="iters")
        wandb.define_metric("running_avg_pck", step_metric="iters")

        wandb.define_metric("epochs")
        wandb.define_metric("trn_loss", step_metric="epochs")
        wandb.define_metric("trn_pck", step_metric="epochs")

        wandb.define_metric("val_loss", step_metric="epochs")
        wandb.define_metric("val_pck", step_metric="epochs")

        if args.criterion == "weak":
            wandb.define_metric("discSelf_loss", step_metric="epochs")
            wandb.define_metric("discCross_loss", step_metric="epochs")
            wandb.define_metric("match_loss", step_metric="epochs")
            
            if args.collect_grad:
                wandb.define_metric("discSelfGrad", step_metric="iters")
                wandb.define_metric("discCrossGrad", step_metric="iters")
                wandb.define_metric("matchGrad", step_metric="iters")                

def save_checkpoint(args, epoch, model, max_pck, optimizer, lr_scheduler):

    save_state = {'model': model.module.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'max_accuracy': max_pck,
                  'lr_scheduler': None,
                  'epoch': epoch,
                  'args': args}
    if lr_scheduler is not None:
        save_state['lr_scheduler'] = lr_scheduler.state_dict()

    save_path = os.path.join(args.logpath, f'ckpt_epoch_{epoch}.pth')
    # Logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    # Logger.info(f"{save_path} saved !!!")

def main(args):
    # ============ 1. Init Logger ... ============
    Logger.initialize(args, training=True)
    args.logpath = Logger.logpath

    rank = dist.get_rank()
    local_rank = args.local_rank
    device = torch.device("cuda:{}".format(local_rank))
    world_size = dist.get_world_size()

    fix_randseed(seed=(0))
    cudnn.benchmark = True
    # cudnn.deterministic = True
    
    # ============ 2. Make Dataloader ... ============
    train_loader, val_loader, aux_val_loader = build_dataloader(args, rank, world_size)

    # ============ 3. Model Intialization ... ============
    assert args.backbone in ["resnet50", "resnet101", "fcn101"], "Unknown backbone"
    Logger.info(f">>>>>>>>>> Creating model:{args.backbone} + {args.selfsup} <<<<<<<<<<")
    n_layers = {"resnet50": 17, "resnet101": 34, "fcn101": 34}
    hyperpixels = list(range(n_layers[args.backbone]))
    model = scot_CAM.SCOT_CAM(args, hyperpixels)
    model.cuda()

    # freeze the backbone
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = False

    # ============ 4. Optimizer, Scheduler, and Loss ... ============
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_scheduler(args, optimizer, len(train_loader), config=None)
    if args.criterion == "weak":
        criterion = WeakDiscMatchLoss(args.temp, args.match_norm_type, args.weak_lambda, args.entropy_func)
    elif args.criterion == "strong_ce":
        criterion = StrongCrossEntropyLoss(args.alpha)
    elif args.criterion == "flow":
        criterion = StrongFlowLoss()
    else:
        raise ValueError("Unknown objective loss")

    # ============ 5. Distributed training ... ============
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=False,
    )
    
    # check # of training params
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger.info(f">>>>>>>>>> number of training params: {n_parameters}")

    # resume the model
    if args.resume or args.selfsup in ["dino", "denseCL"]:
        model_without_ddp = model.module
        max_pck = load_checkpoint(args, model_without_ddp, optimizer, lr_scheduler)
    else:
        max_pck = 0.0

    # ============ 6. Evaluator, Wandb ... ============
    Evaluator.initialize(args.alpha)
    build_wandb(args, rank)

    # ============ 7. Start training ... ============
    log_benchmark = {}
    Logger.info(">>>>>>>>>> Start training")
    args.weak_lambda = torch.FloatTensor(list(map(float, re.findall(r"[-+]?(?:\d*\.*\d+)", args.weak_lambda)))).cuda()
    for epoch in range(args.start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)

        # train
        start_time = time.time()
        trn_loss, trn_pck = train(args, model, criterion, train_loader, optimizer, epoch)
        log_benchmark["trn_loss"] = trn_loss
        log_benchmark["trn_pck"] = trn_pck
        end_train_time = (time.time()-start_time)/60
        # print(epoch, model.module.learner.layerweight.detach().clone().view(-1))

        # validation
        start_time = time.time()
        val_loss, val_pck = validate(
            args, model, criterion, val_loader, epoch, aux_val_loader
        )
        log_benchmark["val_loss"] = val_loss
        log_benchmark["val_pck"] = val_pck
        end_val_time = (time.time()-start_time)/60

        # save model and log results
        if val_pck > max_pck and rank == 0:
            # Logger.save_model(model.module, epoch, val_pck, max_pck)
            # save_checkpoint(args, epoch, model, max_pck, optimizer, lr_scheduler)
            Logger.info('Best Model saved @%d w/ val. PCK: %5.4f -> %5.4f on [%s]\n' % (epoch, max_pck, val_pck, os.path.join(args.logpath, f'ckpt_epoch_{epoch}.pth')))
            max_pck = val_pck

        if args.use_wandb and rank == 0:
            wandb.log({"epochs": epoch})
            wandb.log(log_benchmark)
            
        time_message = (
            ">>>>> Train/Eval %d epochs took:%4.3f + %4.3f = %4.3f"%(epoch + 1, end_train_time, end_val_time, end_train_time+end_val_time)+" minutes"
        )
        Logger.info(time_message)
        if epoch%2 == 0:
            torch.cuda.empty_cache()

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
    # parser.add_argument('--split', type=str, default='trn', help='trn, val, test, old_trn, trn_val') 

    # Training parameters
    parser.add_argument('--criterion', type=str, default='strong_ce', choices=['weak', 'strong_ce', 'flow'])
    parser.add_argument('--lr', type=float, default=0.01) 
    parser.add_argument('--lr_backbone', type=float, default=0.0) 
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--optimizer', type=str, default="sgd", choices=["sgd", "adamw"])
    parser.add_argument('--weight_decay', type=float, default=0.00, help='weight decay (default: 0.00)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (default: 0.9)')
    parser.add_argument("--scheduler", type=str, default="none", choices=['none', 'step', 'cycle', 'cosine'])
    parser.add_argument('--step_size', type=int, default=16, help='hyperparameters for step scheduler')
    parser.add_argument('--step_gamma', type=float, default=0.1, help='hyperparameters for step scheduler')
    
    parser.add_argument("--use_grad_clip", type=boolean_string, nargs="?", default=False)
    parser.add_argument("--grad_clip", type=float, default=0.1, help='gradient clip threshold') 
    parser.add_argument("--use_wandb", type= boolean_string, nargs="?", default=False)
    parser.add_argument("--use_xavier", type= boolean_string, nargs="?", default=False)
    parser.add_argument('--loss_stage', type=str, default="sim", choices=['sim', 'sim_geo', 'votes', 'votes_geo'])
    parser.add_argument('--resume', default='', type=str,help='path to latest checkpoint (default: none)')
    parser.add_argument('--backbone_path', type=str, default='./backbone')
    parser.add_argument('--weight_thres', type=float, default=0.00, help='weight_thres (default: 0.00)')
    parser.add_argument('--select_all', type=float, default=1.01, help='selec all probability (default: 1.0)')
    
    parser.add_argument('--output_image_size', type=str, default='(300)')

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


    parser.add_argument("--match_norm_type", type=str, default="l1", choices=["l1", "linear", "softmax"])
    parser.add_argument('--weak_lambda', type=str, default='[1.0, 1.0, 1.0]')
    parser.add_argument('--temp', type=float, default=0.05, help='softmax-temp for match loss')
    parser.add_argument("--collect_grad", type= boolean_string, nargs="?", default=False)
    parser.add_argument('--entropy_func', type=str, default='info_entropy')
    parser.add_argument("--use_negative", type= boolean_string, nargs="?", default=False)


    # Arguments for distributed data parallel
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int, help='numer of distributed processes')
    parser.add_argument("--local_rank", required=True, type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument("--eval_mode", type= boolean_string, nargs="?", default=False, help='train or test model')
        

    args = parser.parse_args()
    
    init_distributed_mode(args)

    if args.use_negative:
        assert args.criterion == 'weak', "use_negative is only for weak loss"


    if args.use_wandb and args.run_id == '':
        args.run_id = wandb.util.generate_id()
    args.output_image_size = parse_string(args.output_image_size)
    if isinstance(args.output_image_size, int):
        # use target rhf center if only scale the max_side
        args.trg_cen = True
        args.use_batch = False
    else:
        args.trg_cen = False
        args.use_batch = True

    main(args)
