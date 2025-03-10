r"""Beam search for hyperpixel layers"""

import datetime
import argparse
import os

from torch.utils.data import DataLoader
import torch
import time
from data import download
from model import scot_CAM, util, geometry, evaluation
from common import utils
from common.logger import AverageMeter
import logging

# wandb.login()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def test(model, dataloader, args):
    r"""Code for test SCOT"""

    model.eval()
    average_meter = AverageMeter(dataloader.dataset.benchmark, dataloader.dataset.cls)
    for _, batch in enumerate(dataloader):
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
            print("zero_pck: %s_%s"%(batch['src_imname'], batch['trg_imname']))
    
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
    parser.add_argument('--selfsup', type=str, default='supervised', choices=['sup', 'dino', 'denseCL'])
    parser.add_argument('--thres', type=str, default='auto', choices=['auto', 'img', 'bbox'])
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--split', type=str, default='val', help='trn, val, test, old_trn') 

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--pretrained_path', type=str, default='')
    parser.add_argument('--backbone_path', type=str, default='./backbone')

    parser.add_argument('--img_side', type=str, default='(300)')
    parser.add_argument("--use_batch", type= utils.boolean_string, nargs="?", default=False)
    parser.add_argument("--trg_cen", type= utils.boolean_string, nargs="?", default=False)

    # Algorithm parameters
    parser.add_argument('--sim', type=str, default='OTGeo', help='Similarity type: OT, OTGeo, cos, cosGeo')
    parser.add_argument('--exp1', type=float, default=1.0, help='exponential factor on initial cosine cost')
    parser.add_argument('--exp2', type=float, default=1.0, help='exponential factor on final OT scores')
    parser.add_argument('--eps', type=float, default=0.05, help='epsilon for Sinkhorn Regularization')
    parser.add_argument('--classmap', type=int, default=1, help='class activation map: 0 for none, 1 for using CAM')
    parser.add_argument('--cam', type=str, default='', help='activation map folder, empty for end2end computation')

    args = parser.parse_args()

    if args.selfsup in ['dino', 'denseCL']:
        args.backbone_path = os.path.join(args.backbone_path, "%s_%s.pth"%(args.selfsup, args.backbone))

    img_side = util.parse_string(args.img_side)
    if isinstance(img_side, int):
        # use trg_center if only scale the max_side
        args.trg_cen = True
        args.use_batch = False
    else:
        args.trg_cen = False
        args.use_batch = True

    logtime = datetime.datetime.now().__format__('_%m%d_%H%M%S')
    logpath = args.logpath
    logpath = os.path.join('logs', logpath + logtime + '.log')
    util.init_logger(logpath)
    util.log_args(args)

    # 1. CUDA and reproducibility
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    util.fix_randseed(seed=0)
    
    # fmt: on
    # 1. CUDA and reproducibility
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 2. Candidate layers for hyperpixel initialization
    n_layers = {"resnet50": 17, "resnet101": 34, "fcn101": 34}
    hyperpixels = list(range(n_layers[args.backbone]))

    # 3. Model intialization
    model = scot_CAM.SCOT_CAM(
        args.backbone,
        hyperpixels,
        args.benchmark,
        device,
        args.cam,
    )

    model.load_state_dict(torch.load(args.pretrained_path, map_location=device))

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

    # 4. Dataset download & initialization
    num_workers = 16 if torch.cuda.is_available() else 8
    pin_memory = True if torch.cuda.is_available() else False
 
    print("loading Dataset")
    dset = download.load_dataset(args.benchmark, args.datapath, args.thres, device, args.split, args.cam, img_side=img_side, use_resize=True, use_batch=False)
    dataloader = DataLoader(dset, batch_size=1, num_workers=num_workers, pin_memory=pin_memory)
    print("loading finished")

    # 5. Evaluator
    evaluator = evaluation.Evaluator(args.benchmark, device)

    # 7. evaluate SCOT
    print("Eval Start")

    test_pck = test(model, dataloader, args)
    print('avg test pck: ', test_pck)