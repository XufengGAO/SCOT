r"""Beam search for hyperpixel layers"""

import datetime
import argparse
import os

from torch.utils.data import DataLoader
import torch
import time
from PIL import Image
from test_data import download
from model import scot_CAM, util, geometry, evaluation
from common import utils
from common.logger import AverageMeter, Logger
from common.evaluation import Evaluator
import torch.optim as optim
from pprint import pprint
import wandb
from model.base.geometry import Geometry
import logging

# wandb.login()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    logtime = datetime.datetime.now().__format__('_%m%d_%H%M%S')
    logpath = args.logpath
    logpath = os.path.join('logs', logpath + logtime + '.log')
    util.init_logger(logpath)
    util.log_args(args)
    
    # fmt: on
    # 1. CUDA and reproducibility
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 2. Candidate layers for hyperpixel initialization
    n_layers = {"resnet50": 17, "resnet101": 34, "fcn101": 34}
    hyperpixels = list(range(n_layers[args.backbone]))

    # 3. Model intialization
    model = scot_CAM.SCOT_CAM(
        backbone=args.backbone,
        hyperpixel_ids=hyperpixels,
        benchmark=args.benchmark,
        device=device,
        cam=args.cam,
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

    # 4. Dataset download & initialization
    num_workers = 16 if torch.cuda.is_available() else 8
    pin_memory = True if torch.cuda.is_available() else 8
 
    dset = download.load_dataset(args.benchmark, args.datapath, args.thres, device, args.split, args.cam)
    dataloader = DataLoader(dset, batch_size=1, num_workers=0)

    # 5. Evaluator
    evaluator = evaluation.Evaluator(args.benchmark, device)

    # 7. evaluate SCOT
    print("Eval Start")

    train_started = time.time()

    zero_pcks = 0
    srcpt_list = []
    trgpt_list = []
    time_list = []
    PCK_list = []
    for idx, data in enumerate(dataloader):
        threshold = 0.0
        
        # a) Retrieve images and adjust their sizes to avoid large numbers of hyperpixels
        data['src_img'], data['src_kps'], data['src_intratio'] = util.resize(data['src_img'], data['src_kps'][0])
        data['trg_img'], data['trg_kps'], data['trg_intratio'] = util.resize(data['trg_img'], data['trg_kps'][0])
        src_size = data['src_img'].size()
        trg_size = data['trg_img'].size()
        
        if len(args.cam)>0:
            data['src_mask'] = util.resize_mask(data['src_mask'],src_size)
            data['trg_mask'] = util.resize_mask(data['trg_mask'],trg_size)
            data['src_bbox'] = util.get_bbox_mask(data['src_mask'], thres=threshold).to(device)
            data['trg_bbox'] = util.get_bbox_mask(data['trg_mask'], thres=threshold).to(device)
        else:
            data['src_mask'] = None
            data['trg_mask'] = None

        data['alpha'] = args.alpha
        tic = time.time()
        
        data['src_img'] = data['src_img'].unsqueeze(dim=0)
        data['trg_img'] = data['trg_img'].unsqueeze(dim=0)

        with torch.no_grad():
            sim, votes, votes_geo, src_box, trg_box, _ = model(
                data['src_img'],
                data['trg_img'],
                args.sim,
                args.exp1,
                args.exp2,
                args.eps,
                args.classmap,
                src_mask=None,
                trg_mask=None,
                backbone=args.backbone,
            )
            
            
            confidence_ts = votes.squeeze(dim=0)
            conf, trg_indices = torch.max(confidence_ts, dim=1)
            unique, inv = torch.unique(trg_indices, sorted=False, return_inverse=True)
            trgpt_list.append(len(unique))
            srcpt_list.append(len(confidence_ts))

        prd_kps = geometry.predict_test_kps(src_box, trg_box, data['src_kps'], confidence_ts)
        toc = time.time()
        #print(toc-tic)
        time_list.append(toc-tic)
        pair_pck = evaluator.evaluate(prd_kps, data)
        PCK_list.append(pair_pck)
        if pair_pck==0:
            zero_pcks += 1
    
        evaluator.log_result(idx, data=data)

    logging.info('source points:'+str(sum(srcpt_list)*1.0/len(srcpt_list)))
    logging.info('target points:'+str(sum(trgpt_list)*1.0/len(trgpt_list)))
    logging.info('avg running time:'+str(sum(time_list)/len(time_list)))
    evaluator.log_result(len(dset), data=None, average=True)
    logging.info('Total Number of 0.00 pck images:'+str(zero_pcks))