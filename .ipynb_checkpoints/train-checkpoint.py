r"""Beam search for hyperpixel layers"""

import datetime
import argparse
import os

from torch.utils.data import DataLoader
import torch
import time
from PIL import Image
from data import download
from model import scot_CAM, util, geometry, evaluation, scot_CAM2
from model.objective import Objective
from SCOT.common import loss
from common import utils
from common.logger import AverageMeter, Logger
from common.evaluation import Evaluator
import torch.optim as optim
from pprint import pprint
import wandb
from model.base.geometry import Geometry
from model import util
# wandb.login()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_knn(db_vectors, qr_vectors):
    r"""Finds K-nearest neighbors (Euclidean distance)"""
    # print("knn", db_vectors.unsqueeze(1).size(), qr_vectors.size())
    # print("knn", db_vectors[-3])
    # (3600, 40, 2), repeated centers for each rep field of each hyperpixel
    db = db_vectors.unsqueeze(1).repeat(1, qr_vectors.size(0), 1)

    # (3600, 40, 2), repeated 40 keypoints
    qr = qr_vectors.unsqueeze(0).repeat(db_vectors.size(0), 1, 1)
    dist = (db - qr).pow(2).sum(2).pow(0.5).t() # (40, 3600)
    # keypoint to each center
    # print("dist", dist.size())
    _, nearest_idx = dist.min(dim=1) #  hyperpixel idx for each keypoint
    # print("nea_idx", nearest_idx.size())
    return nearest_idx

def match_idx(kpss, n_ptss, rf_center):
    r"""Samples the nearst feature (receptive field) indices"""
    max_pts = 40
    batch = len(kpss)
    nearest_idxs = torch.zeros((batch, max_pts), dtype=torch.int32).to(device)
    for idx, (kps, n_pts) in enumerate(zip(kpss, n_ptss)):
        nearest_idx = find_knn(rf_center, kps[:,:n_pts].t())
        nearest_idxs[idx, :n_pts] = nearest_idx
        # nearest_idxs.append(nearest_idx.unsqueeze(0))
    # nearest_idxs = torch.cat(nearest_idxs, dim=0)

    return nearest_idxs

def train(epoch, model, dataloader, strategy, optimizer, training, args):
    r"""Code for training SCOT"""
    if training == "train":
        model.train()
    else:
        model.eval()

    average_meter = AverageMeter(dataloader.dataset.benchmark, dataloader.dataset.cls)
    total_steps = len(dataloader)

    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        iters = step + epoch * total_steps
        
        src_img, trg_img = strategy.get_image_pair(batch) 

        if "src_mask" in batch.keys():
            batch["src_mask"] = batch["src_mask"].to(device)
            batch["trg_mask"] = batch["trg_mask"].to(device)
        else:
            batch["src_mask"] = None
            batch["trg_mask"] = None

        sim, votes, votes_geo, src_box, trg_box, feat_size, src_center, trg_center, src_hf, trg_hf = model(
            src_img,
            trg_img,
            args.sim,
            args.exp1,
            args.exp2,
            args.eps,
            args.classmap,
            batch["src_mask"],
            batch["trg_mask"],
            args.backbone,
            training,
            args.trg_cen
        )

        # print(sim.size(), votes.size(), votes_geo.size(), src_box.size(), trg_box.size())
        # print("model result", confidence_ts.size(), src_box.size(), trg_box.size())
        # model result torch.Size([4, 4096, 4096]) torch.Size([4096, 4]) torch.Size([4096, 4])
        # print("geometry", Geometry.rf_center.size())

        model_outputs = [sim, votes, votes_geo]
        with torch.no_grad():
            batch['src_kps'] = batch['src_kps'].to(device)
            batch['n_pts'] = batch['n_pts'].to(device)
            batch['trg_kps'] = batch['trg_kps'].to(device)
            batch['pckthres'] = batch['pckthres'].to(device)
            
            # for cross-entropy loss
            if args.loss == "strong_ce":
                batch['src_kpidx'] = match_idx(batch['src_kps'], batch['n_pts'], src_center)
                if args.trg_cen: # use trg_center
                    batch['trg_kpidx'] = match_idx(batch['trg_kps'], batch['n_pts'], trg_center)
                else:
                    batch['trg_kpidx'] = match_idx(batch['trg_kps'], batch['n_pts'], src_center)

            # for flow loss
            if args.loss == "flow":
                batch['flow'] = Geometry.KpsToFlow(batch['src_kps'], batch['trg_kps'], batch['n_pts'])

            prd_kps_list = []
            eval_result_list = []

            for corr in model_outputs:
                prd_kps = geometry.predict_kps(
                    src_box,
                    trg_box,
                    batch["src_kps"],
                    batch["n_pts"],
                    strategy.get_correlation(corr),
                ) # [2,2,40]
                prd_kps_list.append(prd_kps)

                eval_result = Evaluator.evaluate(prd_kps, batch, args.loss) # return dict results
                # print(len(eval_result['pck']))
                eval_result_list.append(eval_result)
        
        assert args.loss_stage in ["sim", "votes", "votes_geo"], "Unrecognized loss stage"
        loss_dict = {"sim":0, "votes":1, "votes_geo":2}
        idx = loss_dict[args.loss_stage]

        if args.loss == "strong_ce":
            loss = strategy.compute_loss(model_outputs[idx], eval_result_list[idx], batch)
        elif args.loss == "flow":
            loss = strategy.compute_loss(model_outputs[idx], batch['flow'].to(device), feat_size)
        elif args.loss in ["softwarp", "hardwarp"]:
            loss = strategy.compute_loss(model_outputs[idx], src_hf, trg_hf, args.loss)


        if training == "train":
            loss.backward()
            if args.use_grad_clip:
                torch.nn.utils.clip_grad.clip_grad_value_(model.parameters(), args.grad_clip)
            optimizer.step()

        # log pck, loss
        average_meter.update(
            eval_result_list,
            batch["pair_class"],
            loss.item(),
        )
        
        # log batch loss, batch pck    
        if training and (step % 60 == 0):
            average_meter.write_process(step, len(dataloader), epoch)

        # log running step loss 
        if False and (step % 100 == 0):
            running_avg_loss = utils.mean(average_meter.loss_buffer)
            running_avg_pck_sim = utils.mean(average_meter.buffer['sim']["pck"])
            running_avg_pck_votes = utils.mean(average_meter.buffer['votes']["pck"])
            running_avg_pck_votes_geo = utils.mean(average_meter.buffer['votes_geo']["pck"])

            grad_ratio = (torch.norm(
                model.learner.layerweight.grad.detach().clone().view(-1)
            ) / (
                torch.norm(model.learner.layerweight.detach().clone().view(-1)) + 1e-20
            )).item()
            
            # print(type(running_avg_loss))
            # print(type(running_avg_pck_sim))
            # print(type(running_avg_pck_votes))
            # print(type(running_avg_pck_votes_geo))
            # print(type(grad_ratio))

            if args.use_wandb:
                wandb.log({"iters": iters})
                wandb.log(
                    {
                        "grad_ratio": grad_ratio,
                        "running_trn_avg_loss": running_avg_loss,
                        "running_avg_pck_sim": running_avg_pck_sim,
                        "running_avg_pck_votes": running_avg_pck_votes,
                        "running_avg_pck_votes_geo": running_avg_pck_votes_geo
                    },
                )

        if (step % 240 == 0):
            # 1. Draw weight map
            weight_map_path = os.path.join(Logger.logpath, "weight_map")
            os.makedirs(weight_map_path, exist_ok=True)
            weight_pth = utils.draw_weight_map(
                model.learner.layerweight.detach().clone().view(-1),
                epoch,
                step,
                weight_map_path,
            )
            # wandb.log({"Sigmoid weight":wandb.Image(fig)})
            # plt.close(fig)

            # 2. Draw matches
            min_pck, min_pck_idx = torch.tensor(eval_result_list[2]["pck"]).min(dim=0)
            src_img = dataloader.dataset.get_image(batch["src_imname"][min_pck_idx])
            trg_img = dataloader.dataset.get_image(batch["trg_imname"][min_pck_idx])
            draw_match_path = os.path.join(Logger.logpath, "draw_match")
            os.makedirs(draw_match_path, exist_ok=True)
            match_pth = utils.draw_matches_on_image(
                epoch,
                step,
                min_pck_idx.item(),
                min_pck,
                src_img,
                trg_img,
                batch,
                pred_trg_kps=prd_kps_list[2][min_pck_idx],
                origin=True,
                color_ids=eval_result_list[2]["pck_ids"][min_pck_idx],
                draw_match_path=draw_match_path,
            )
            del min_pck, min_pck_idx
            
            if args.use_wandb:
                wandb.log(
                    {
                        "weight_map": wandb.Image(Image.open(weight_pth).convert("RGB")),
                        "match_map": wandb.Image(Image.open(match_pth).convert("RGB"))
                    }
                )
                
        del sim, votes, votes_geo, src_box, trg_box, feat_size, src_center, trg_center, src_hf, trg_hf
        del model_outputs, loss, eval_result_list, prd_kps_list

    # 2.5
    if args.split == "trnval" and args.use_wandb:
        weight_list =  model.learner.layerweight.view(-1).sigmoid().tolist()
        weight_dict = dict()
        for i, weight in enumerate(weight_list):
            weight_dict['w%d'%(i)] = weight
        wandb.log(weight_dict)
            
    
    # 3. Draw class pck
    if (epoch % 2)==0:
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

    # log epoch loss, epoch pck
    average_meter.write_result("Training" if training == "train" else "Validation", epoch)

    avg_loss = utils.mean(average_meter.loss_buffer)
    avg_pck = {'sim':0, 'votes':0, 'votes_geo':0}
    for key, value in avg_pck.items():
        avg_pck[key] = utils.mean(average_meter.buffer[key]["pck"])

    return avg_loss, avg_pck

def test(model, dataloader, args):
    r"""Code for test SCOT"""

    model.eval()
    average_meter = AverageMeter(dataloader.dataset.benchmark, dataloader.dataset.cls)
    for step, batch in enumerate(dataloader):
        batch['src_img'] = batch['src_img'].to(device)
        batch['trg_img'] = batch['trg_img'].to(device)
        
        if "src_mask" in batch.keys():
            src_mask = batch["src_mask"].to(device)
            trg_mask = batch["trg_mask"].to(device)
        else:
            src_mask = None
            trg_mask = None

        votes_geo, src_box, trg_box = model(
            batch['src_img'],
            batch['trg_img'],
            args.sim,
            args.exp1,
            args.exp2,
            args.eps,
            args.classmap,
            src_mask,
            trg_mask,
            args.backbone,
            training="test",
        )
        batch['src_kps'] = batch['src_kps'].to(device)

        prd_kps = geometry.predict_test_kps(src_box, trg_box, batch['src_kps'][0], votes_geo[0])
        pair_pck = average_meter.eval_pck(prd_kps, batch, args.alpha)

        if pair_pck==0:
            Logger.info("zero_pck: %s_%s"%(batch['src_imname'], batch['trg_imname']))
    
    avg_pck =  average_meter.log_pck()
    del votes_geo, src_box, trg_box
    return avg_pck

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
    parser.add_argument('--loss_stage', type=str, default="sim", choices=["sim", "votes", "votes_geo"])
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
    parser.add_argument('--eps', type=float, default=0.05, help='epsilon for Sinkhorn Regularization')
    parser.add_argument('--classmap', type=int, default=0, help='class activation map: 0 for none, 1 for using CAM')
    parser.add_argument('--cam', type=str, default='', help='activation map folder, empty for end2end computation')
    # default is the value that the attribute gets when the argument is absent. const is the value it gets when given.

    parser.add_argument('--run_id', type=str, default='', help='run_id')
    parser.add_argument('--wandb_proj', type=str, default='', help='wandb project name')

    args = parser.parse_args()

    if args.selfsup in ['dino', 'denseCL']:
        args.backbone_path = os.path.join(args.backbone_path, "%s_%s.pth"%(args.selfsup, args.backbone))

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
        
    Logger.initialize(args)
    
    # fmt: on
    # 1. CUDA and reproducibility
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    util.fix_randseed(seed=0)

    # 2. Candidate layers
    n_layers = {"resnet50": 17, "resnet101": 34, "fcn101": 34}
    hyperpixels = list(range(n_layers[args.backbone]))

    # 3. Model intialization
    if args.use_scot2:
        model = scot_CAM2.SCOT_CAM(
            args.backbone,
            hyperpixels,
            args.benchmark,
            device,
            args.cam,
            args.use_xavier,
            weight_thres=args.weight_thres,
            select_all=args.select_all
        )
    else:
        model = scot_CAM.SCOT_CAM(
            args.backbone,
            hyperpixels,
            args.benchmark,
            device,
            args.cam,
            args.use_xavier,
            args.weight_thres,
            args.select_all
        )

    if args.resume:
        model.load_state_dict(torch.load(args.resume_path, map_location=device))

    if args.selfsup in ['dino', 'denseCL']:
        pretrained_backbone = torch.load(args.backbone_path, map_location=device)
        backbone_keys = list(model.backbone.state_dict().keys())
            
        if 'state_dict' in pretrained_backbone:
            model.load_backbone(pretrained_backbone['state_dict'])
            load_keys = list(pretrained_backbone['state_dict'].keys())
        else:
            model.load_backbone(pretrained_backbone)
            load_keys = list(pretrained_backbone.keys())
        missing_keys = [i for i in backbone_keys if i not in load_keys]
        print(missing_keys)

    # 4. Objective (cross-entropy loss) and Optimizer
    Objective.initialize(target_rate=0.5, alpha=args.alpha)
    if args.loss == "weak":
        strategy = loss.WeakLoss()
    elif args.loss == "strong_ce":
        strategy = loss.StrongCELoss()
    elif args.loss == "flow":
        strategy = loss.StrongFlowLoss()
    else:
        raise ValueError("Unrecognized objective loss")

    assert args.optimizer in ["sgd", "adam"], "Unrecognized model type"
    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.use_wandb:
        wandb_name = "%.e_%s_%s_%s"%(args.lr, args.loss_stage, args.loss, args.optimizer)
        if args.use_scheduler:
            wandb_name += "_%s"%(args.scheduler)
        if args.optimizer == "sgd":
            wandb_name = wandb_name + "_m%.2f"%(args.momentum)
        # if args.trg_cen:
        wandb_name = wandb_name + "_bsz%d"%(args.batch_size)
        
        if args.use_scot2:
            wandb_name = wandb_name + "_scot2"
        # if args.selfsup in ['dino', 'denseCL']:
        wandb_name = wandb_name + "_%s_%s_%s_alp%.2f"%(args.selfsup, args.backbone, args.split, args.alpha)

        run = wandb.init(project=args.wandb_proj, config=args, id=args.run_id, resume="allow", name=wandb_name)
        # wandb.watch(model.learner, log="all", log_freq=100)
        wandb.define_metric("iters")
        wandb.define_metric("grad_ratio", step_metric="iters")
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
     
        wandb.define_metric("test_pck_sim", step_metric="epochs")
        wandb.define_metric("test_pck_votes", step_metric="epochs")
        wandb.define_metric("test_pck_votes_geo", step_metric="epochs")
        
        if args.split == "trnval":
            for i in range(n_layers[args.backbone]):
                wandb.define_metric('w%d'%(i), step_metric="epochs")
        
    # 5. Dataset download & initialization
    num_workers = 16 if torch.cuda.is_available() else 8
    pin_memory = True if torch.cuda.is_available() else False
    
    print("loading Dataset")
    
    trn_ds = download.load_dataset(args.benchmark, args.datapath, args.thres, device, args.split, args.cam, img_side=img_side, use_resize=True, use_batch=args.use_batch)
    trn_dl = DataLoader(dataset=trn_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    
    if args.split not in ["val", "trnval"]:
        val_ds = download.load_dataset(args.benchmark, args.datapath, args.thres, device, "val", args.cam, img_side=img_side, use_resize=True, use_batch=args.use_batch)
        val_dl = DataLoader(dataset=val_ds, batch_size=args.batch_size, num_workers=num_workers, pin_memory=pin_memory)
    
    if args.benchmark in ['pfpascal']:
        test_ds = download.load_dataset(args.benchmark, args.datapath, args.thres, device, "test", args.cam, img_side=img_side, use_resize=True, use_batch=False)
        test_dl = DataLoader(dataset=test_ds, batch_size=1, num_workers=num_workers, pin_memory=pin_memory)
    print("loading finished")
    
    scheduler = None
    if args.use_scheduler:
        assert args.scheduler in ["cycle", "step", "cosin"], "Unrecognized model type" 
        if args.scheduler == "cycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, epochs=args.epochs, steps_per_epoch=len(trn_dl))
        elif args.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.step_gamma)

    # 6. Evaluator
    Evaluator.initialize(args.alpha)

    # 7. Train SCOT
    best_val_pck = float("-inf")
    log_benchmark = {}
    print("Training Start")
    train_started = time.time()
    
    if args.use_wandb:
        wandb.log({'epochs':-2})
    with torch.no_grad():
        test_pck = test(model, test_dl, args)
        log_benchmark["test_pck"] = test_pck
    if args.use_wandb:
        wandb.log({'epochs':-1})
        wandb.log(log_benchmark)
        
    for epoch in range(args.start_epoch, args.epochs):

        # training
        torch.cuda.empty_cache()
        trn_loss, trn_pck = train(epoch, model, trn_dl, strategy, optimizer, training="train", args=args)
        log_benchmark["trn_loss"] = trn_loss
        log_benchmark["trn_pck_sim"] = trn_pck['sim']
        log_benchmark["trn_pck_votes"] = trn_pck['votes']
        log_benchmark["trn_pck_votes_geo"] = trn_pck['votes_geo']
        
        # validation
        if args.split not in ["val", "trnval"]:
            with torch.no_grad():
                val_loss, val_pck = train(epoch, model, val_dl, strategy, optimizer, training="val", args=args)
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
            
        save_e = 1 if args.split == "trnval" else 5
        if (epoch%save_e)==0:
            Logger.save_epoch(model, epoch, model_pck["votes_geo"])
            
        if model_pck["votes_geo"] > best_val_pck:
            old_best_val_pck = best_val_pck
            best_val_pck = model_pck["votes_geo"]
            Logger.save_model(model, epoch, best_val_pck, old_best_val_pck)

        # testing
        if args.benchmark in ['pfpascal']:
            with torch.no_grad():
                test_pck = test(model, test_dl, args)
                log_benchmark["test_pck"] = test_pck

        if args.use_wandb:
            wandb.log({'epochs':epoch})
            wandb.log(log_benchmark)

        time_message = 'Training %d epochs took:%4.3f\n' % (epoch+1, (time.time()-train_started)/60) + ' minutes'
        Logger.info(time_message)
        
        torch.cuda.empty_cache()

    Logger.info("==================== Finished training ====================")