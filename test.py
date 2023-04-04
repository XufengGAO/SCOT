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
from common import supervision as sup
from common import utils
from common.logger import AverageMeter, Logger
from common.evaluation import Evaluator
import torch.optim as optim
from pprint import pprint
import wandb
from model.base.geometry import Geometry

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

def match_idx(kpss, n_ptss):
    r"""Samples the nearst feature (receptive field) indices"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nearest_idxs = []
    max_pts = 40
    range_ts = torch.arange(max_pts).to(device)
    for kps, n_pts in zip(kpss, n_ptss):
        nearest_idx = find_knn(Geometry.rf_center, kps.t())
        nearest_idx -= (range_ts >= n_pts).to(device).long()
        nearest_idxs.append(nearest_idx.unsqueeze(0))
    nearest_idxs = torch.cat(nearest_idxs, dim=0)

    return nearest_idxs.to(device)

def eval(model, dataloader, args):
    r"""Code for training SCOT"""

    model.eval()
    average_meter = AverageMeter(dataloader.dataset.benchmark, cls=dataloader.dataset.cls)
    for step, batch in enumerate(dataloader):
        src_img, trg_img = batch['src_img'], batch['trg_img']
        src_img = src_img.to(device)
        trg_img = trg_img.to(device)
        
        if "src_mask" in batch.keys():
            src_mask = batch["src_mask"].to(device)
            trg_mask = batch["trg_mask"].to(device)
        else:
            src_mask = None
            trg_mask = None

        sim, votes, votes_geo, src_box, trg_box, _ = model(
            src_img,
            trg_img,
            args.sim,
            args.exp1,
            args.exp2,
            args.eps,
            args.classmap,
            src_mask,
            trg_mask,
            args.backbone,
        )

        model_outputs = [sim, votes, votes_geo]
        with torch.no_grad():

            batch['src_kps'] = batch['src_kps'].to(device)
            batch['n_pts'] = batch['n_pts'].to(device)
            batch['trg_kps'] = batch['trg_kps'].to(device)
            batch['pckthres'] = batch['pckthres'].to(device)
            
            batch['src_kpidx'] = match_idx(batch['src_kps'], batch['n_pts'])
            batch['trg_kpidx'] = match_idx(batch['trg_kps'], batch['n_pts'])

            prd_kps_list = []
            eval_result_list = []
            for corr in model_outputs:
                prd_kps = geometry.predict_kps(
                    src_box,
                    trg_box,
                    batch["src_kps"],
                    batch["n_pts"],
                    corr.detach().clone(),
                )
                prd_kps_list.append(prd_kps)
                eval_result = Evaluator.evaluate(prd_kps, batch)
                eval_result_list.append(eval_result)

        # log pck, loss
        average_meter.update(
            eval_result_list,
            batch["pair_class"],
        )

        average_meter.write_process(step, len(dataloader))

    # 3. Draw class pck
    if args.use_wandb:
        draw_class_pck_path = os.path.join(Logger.logpath, "draw_class_pck")
        os.makedirs(draw_class_pck_path, exist_ok=True)
        class_pth = utils.draw_class_pck(
            average_meter.sel_buffer["votes_geo"], draw_class_pck_path
        )
        if args.use_wandb:
            wandb.log(
                {
                    "class_pck": wandb.Image(Image.open(class_pth).convert("RGB")),
                }
            )

    average_meter.write_result("Evaluation")

    avg_pck = {'sim':0, 'votes':0, 'votes_geo':0}
    for key, value in avg_pck.items():
        avg_pck[key] = utils.mean(average_meter.buffer[key]["pck"])

    return avg_pck


if __name__ == "__main__":
    # Arguments parsing
    # fmt: off
    parser = argparse.ArgumentParser(description="SCOT Training Script")

    parser.add_argument('--gpu', type=str, default='0', help='GPU id')
    parser.add_argument('--datapath', type=str, default='./Datasets_SCOT') 
    parser.add_argument('--benchmark', type=str, default='pfpascal')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--selfsup', type=str, default='supervised', choices=['sup', 'dino', 'denseCL'])
    parser.add_argument('--thres', type=str, default='auto', choices=['auto', 'img', 'bbox'])
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--split', type=str, default='val', help='trn, val, test, old_trn') 

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--use_wandb", type= utils.boolean_string, nargs="?", default=False)
    parser.add_argument('--pretrained_path', type=str, default='')
    parser.add_argument('--backbone_path', type=str, default='./backbone')
    parser.add_argument('--weight_thres', type=float, default=0.05,help='weight_thres (default: 0.05)')

    # Algorithm parameters
    parser.add_argument('--sim', type=str, default='OTGeo', help='Similarity type: OT, OTGeo, cos, cosGeo')
    parser.add_argument('--exp1', type=float, default=1.0, help='exponential factor on initial cosine cost')
    parser.add_argument('--exp2', type=float, default=1.0, help='exponential factor on final OT scores')
    parser.add_argument('--eps', type=float, default=0.05, help='epsilon for Sinkhorn Regularization')
    parser.add_argument('--classmap', type=int, default=1, help='class activation map: 0 for none, 1 for using CAM')
    parser.add_argument('--cam', type=str, default='', help='activation map folder, empty for end2end computation')

    parser.add_argument('--run_id', type=str, default='', help='run_id')

    args = parser.parse_args()

    if args.selfsup in ['dino', 'denseCL']:
        args.backbone_path = os.path.join(args.backbone_path, "%s_%s.pth"%(args.selfsup, args.backbone))
        args.classmap = 0

    if args.use_wandb and args.run_id == '':
        args.run_id = wandb.util.generate_id()

    Logger.initialize(args, training=False)
    
    # fmt: on
    # 1. CUDA and reproducibility
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 2. Candidate layers for hyperpixel initialization
    n_layers = {"resnet50": 17, "resnet101": 34, "fcn101": 34}
    hyperpixels = list(range(n_layers[args.backbone]))

    # 3. Model intialization
    img_side = (256, 256)
    model = scot_CAM.SCOT_CAM(
        backbone=args.backbone,
        hyperpixel_ids=hyperpixels,
        benchmark=args.benchmark,
        device=device,
        cam=args.cam,
        img_side=img_side,
        weight_thres=args.weight_thres,
        training=False
    )


    model.load_state_dict(torch.load(args.pretrained_path, map_location=device))

    if args.selfsup in ['dino', 'denseCL']:
        # print(model.backbone.conv1.weight[1,:,:2,:2])
        # print(model.backbone.fc.weight[:2,:5])
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
        # print(model.backbone.conv1.weight[1,:,:2,:2])
        # print(model.backbone.fc.weight[:2,:5])

    if args.use_wandb:
        run = wandb.init(project="SCOT", config=args, id=args.run_id, resume="allow")
        # wandb.watch(model.learner, log="all", log_freq=100)
        wandb.define_metric("epochs")
        
        wandb.define_metric("evaluate_pck_sim", step_metric="epochs")
        wandb.define_metric("evaluate_pck_votes", step_metric="epochs")
        wandb.define_metric("evaluate_pck_votes_geo", step_metric="epochs")

    # 5. Dataset download & initialization
    num_workers = 16 if torch.cuda.is_available() else 8
    pin_memory = True if torch.cuda.is_available() else 8
 
    eva_ds = download.load_dataset(
        args.benchmark, args.datapath, args.thres, device, args.split, img_side=img_side
    )
    eva_dl = DataLoader(dataset=eva_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    # 6. Evaluator
    Evaluator.initialize(args.benchmark, args.alpha)

    # 7. Train SCOT
    best_val_pck = float("-inf")
    log_benchmark = {}

    print("Eval Start")

    train_started = time.time()
    with torch.no_grad():
        eval_pck = eval(model, eva_dl, args=args)
    time_message = 'Eval took:%4.3f\n' % ((time.time()-train_started)/60) + ' minutes'
    Logger.info(time_message)

    Logger.info("==================== Finished evaluation ====================")
