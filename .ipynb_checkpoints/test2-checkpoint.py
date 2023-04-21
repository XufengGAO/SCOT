r"""Beam search for hyperpixel layers"""

import datetime
import argparse
import os

from torch.utils.data import DataLoader
import torch
import time
from test_data import download
from model import scot_CAM, util, geometry, evaluation
from common import utils
import logging

import matplotlib.patches as patches
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# wandb.login()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_concat_h(im1, im2):
    max_height = max(im1.height, im2.height)
    dst = Image.new('RGB', (im1.width + im2.width, max_height), color="white")
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def draw_matches_on_image(match_pck, src_img, trg_img, batch, pred_trg_kps=None, origin=False, color_ids=None, draw_match_path=None):
    r"""Draw keypoints on image
    
    Args:
        match_idx: sample id in one batch
        src_img, trg_img: The original PIL.Image object

    """
    
    n_pts = batch['n_pts']

    # 2. Check image rescale
    if origin:
        src_img = src_img.resize((int(src_img.width/batch['src_intratio']), int(src_img.height/batch['src_intratio'])), Image.Resampling.BILINEAR)
        trg_img = trg_img.resize((int(src_img.width/batch['trg_intratio']), int(src_img.height/batch['trg_intratio'])), Image.Resampling.BILINEAR)
        
        src_ratio = batch['src_intratio'] # wxh
        trg_ratio = batch['trg_intratio']
    else:
        src_ratio = 1.0
        trg_ratio = 1.0

        
        
    # 3. Check kps
    src_kps = (batch['src_kps'][:,:n_pts].cpu().numpy() / src_ratio)

    if pred_trg_kps is not None:
        pred_trg_kps = (pred_trg_kps[:,:n_pts].cpu().numpy() / trg_ratio)
    else:
        pred_trg_kps = batch['trg_kps'][:,:n_pts].cpu().numpy() / trg_ratio
    
    trg_kps = batch['trg_kps'][:,:n_pts].cpu().numpy() / trg_ratio
    
    # 4. Check bounding box
    src_bbox = batch['src_bbox'][0].cpu().numpy()
    trg_bbox = batch['trg_bbox'][0].cpu().numpy()
 
    src_bbox_start = (src_bbox[0]/src_ratio, src_bbox[1]/src_ratio)
    src_bbox_w, src_bbox_h = (src_bbox[2] - src_bbox[0])/src_ratio, (src_bbox[3] - src_bbox[1])/src_ratio

    trg_bbox_start = ((trg_bbox[0]/trg_ratio + src_img.width), trg_bbox[1]/trg_ratio)
    trg_bbox_w, trg_bbox_h = (trg_bbox[2] - trg_bbox[0])/trg_ratio, (trg_bbox[3] - trg_bbox[1])/trg_ratio

    src_rect = patches.Rectangle(src_bbox_start, src_bbox_w, src_bbox_h, linewidth=2, edgecolor='b', facecolor='none')
    trg_rect = patches.Rectangle(trg_bbox_start, trg_bbox_w, trg_bbox_h, linewidth=2, edgecolor='b', facecolor='none')

    # 5. Concatenate images horinzontally
    con_img = get_concat_h(src_img, trg_img)
    con_img = np.array(con_img)

    # 6. Draw the matches
    fig, ax = plt.subplots()
    ax.imshow(con_img)

    

    colors = ['red', 'green', 'yellow']
    if color_ids is None:
        color_ids = torch.ones(n_pts, dtype=torch.uint8)

    for pt_idx in range(n_pts):
        
        ax.plot(src_kps[0,pt_idx], src_kps[1,pt_idx], marker='o', color=colors[color_ids[pt_idx]])
        ax.plot(pred_trg_kps[0,pt_idx] + src_img.width, pred_trg_kps[1,pt_idx], marker='o', color=colors[color_ids[pt_idx]])
        ax.plot([src_kps[0,pt_idx], pred_trg_kps[0,pt_idx] + src_img.width], [src_kps[1,pt_idx], pred_trg_kps[1,pt_idx]], color=colors[color_ids[pt_idx]], linewidth=2)
        if color_ids[pt_idx] == 0:
            ax.plot(trg_kps[0,pt_idx] + src_img.width, trg_kps[1,pt_idx], marker='x', color=colors[2])

    ax.add_patch(src_rect)
    ax.add_patch(trg_rect)

    img_name = "{}-{}.png".format(batch["src_imname"][0][:-4], batch["trg_imname"][0][:-4])
    ax.set_title("pck=%.3f" % (match_pck))
    fig.tight_layout()

    match_pth = os.path.join(draw_match_path, img_name)
    plt.axis('off')
    fig.savefig(match_pth, bbox_inches='tight')
    plt.close('all')

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

    logpath = os.path.join('logs', "test_%s_%s_%.2f"%(args.selfsup, args.backbone, args.alpha))
    os.makedirs(logpath, exist_ok=True)
    logger = os.path.join(logpath, 'test.log')
    util.init_logger(logger)
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
    dset = download.load_dataset(args.benchmark, args.datapath, args.thres, device, args.split, args.cam)
    dataloader = DataLoader(dset, batch_size=1, num_workers=0, pin_memory=False)
    print("loading finished")

    # 7. evaluate SCOT
    print("Eval Start")
    evaluator = evaluation.Evaluator(args.benchmark, device)

    zero_pcks = 0
    srcpt_list = []
    trgpt_list = []
    time_list = []
    PCK_list = []
    for idx, data in enumerate(dataloader):
        data['src_kps'] = data['src_kps'].to(device)
        data['trg_kps'] = data['trg_kps'].to(device)

        data['src_img'] = data['src_img'].to(device)
        data['trg_img'] = data['trg_img'].to(device)

        src_img = data['src_img'][0].clone()
        trg_img = data['trg_img'][0].clone()


        img_name = os.path.join(dset.img_path, data['src_imname'][0])
        src_img = Image.open(img_name).convert('RGB') # WxH

        img_name = os.path.join(dset.img_path, data['trg_imname'][0])
        trg_img = Image.open(img_name).convert('RGB') # WxH   
        
        data['src_bbox'] = data['src_bbox'].to(device)
        data['trg_bbox'] = data['trg_bbox'].to(device)
        data['pckthres'] = data['pckthres'].to(device)
        
        threshold = 0.0

        # a) Retrieve images and adjust their sizes to avoid large numbers of hyperpixels
        data['src_img'], data['src_kps'], data['src_intratio'] = util.resize(data['src_img'], data['src_kps'][0])
        data['trg_img'], data['trg_kps'], data['trg_intratio'] = util.resize(data['trg_img'], data['trg_kps'][0])
        
        # resize image
        src_img = src_img.resize((data['src_img'].size()[2], data['src_img'].size()[1]), Image.Resampling.BILINEAR)
        trg_img = trg_img.resize((data['trg_img'].size()[2], data['trg_img'].size()[1]), Image.Resampling.BILINEAR)
        
        # resize box
        
        data['src_bbox'] = data['src_bbox'] * data['src_intratio']
        data['src_bbox'] = data['src_bbox'] * data['trg_intratio']
                
        data['n_pts'] = data['src_kps'].size()[1]

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
            votes_geo, src_box, trg_box = model(
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
                training="test",
            )

            confidence_ts = votes_geo.squeeze(dim=0)
            conf, trg_indices = torch.max(confidence_ts, dim=1)
            unique, inv = torch.unique(trg_indices, sorted=False, return_inverse=True)
            trgpt_list.append(len(unique))
            srcpt_list.append(len(confidence_ts))

        prd_kps = geometry.predict_test_kps(src_box, trg_box, data['src_kps'], confidence_ts)
        toc = time.time()
        #print(toc-tic)
        time_list.append(toc-tic)
        pair_pck, correct_pts = evaluator.evaluate(prd_kps, data)

        draw_matches_on_image(pair_pck, src_img, trg_img, data, prd_kps, color_ids=correct_pts, draw_match_path=logpath)

        PCK_list.append(pair_pck)
        if pair_pck==0:
            zero_pcks += 1

        evaluator.log_result(idx, data=data)

    logging.info('source points:'+str(sum(srcpt_list)*1.0/len(srcpt_list)))
    logging.info('target points:'+str(sum(trgpt_list)*1.0/len(trgpt_list)))
    logging.info('avg running time:'+str(sum(time_list)/len(time_list)))
    evaluator.log_result(len(dset), data=None, average=True)
    logging.info('Total Number of 0.00 pck images:'+str(zero_pcks))