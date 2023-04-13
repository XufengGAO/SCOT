r"""Beam search for hyperpixel layers"""

import argparse
import os
from torch.utils.data import DataLoader
import torch
from data import download
from model import scot_CAM, util
from data.mask import MaskDataset
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    
    # Arguments parsing
    # fmt: off
    parser = argparse.ArgumentParser(description="SCOT Training Script")

    parser.add_argument('--gpu', type=str, default='0', help='GPU id')
    parser.add_argument('--datapath', type=str, default='./Datasets_SCOT') 
    parser.add_argument('--benchmark', type=str, default='pfpascal')
    parser.add_argument('--backbone', type=str, default='resnet101')
    parser.add_argument('--thres', type=str, default='auto', choices=['auto', 'img', 'bbox'])
    parser.add_argument('--split', type=str, default='trn', help='trn, val, test, old_trn') 
    parser.add_argument('--classmap', type=int, default=0, help='class activation map: 0 for none, 1 for using CAM')
    parser.add_argument('--cam', type=str, default='', help='activation map folder, empty for end2end computation')
    # default is the value that the attribute gets when the argument is absent. const is the value it gets when given.

    args = parser.parse_args()
    
    # fmt: on
    # 1. CUDA and reproducibility
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    util.fix_randseed(seed=0)

    # 2. Candidate layers for hyperpixel initialization
    n_layers = {"resnet50": 17, "resnet101": 34, "fcn101": 34}
    hyperpixels = list(range(n_layers[args.backbone]))

    # 3. Model intialization
    model = scot_CAM.SCOT_CAM(
        args.backbone,
        hyperpixels,
        args.benchmark,
        device,
    )
    model.eval()

    # 5. Dataset download & initialization
    num_workers = 16 if torch.cuda.is_available() else 8
    pin_memory = True if torch.cuda.is_available() else False

    mask_ds = MaskDataset(args.benchmark, args.datapath, device, "mask", imside=(200,300), use_resize=True)
    mask_dl = DataLoader(dataset=mask_ds, batch_size=1, num_workers=1, pin_memory=pin_memory)

    transform = T.ToPILImage()
    f0_folder = os.path.join("./Datasets_SCOT/PF-PASCAL/mask", args.backbone,'200_300','f0')
    os.makedirs(f0_folder, exist_ok=True)
    f1_folder = os.path.join("./Datasets_SCOT/PF-PASCAL/mask", args.backbone,'200_300','f1')
    os.makedirs(f1_folder, exist_ok=True)

    with torch.no_grad():
        for step, batch in tqdm(enumerate(mask_dl)):
            img_imname, src_img = batch['img_imname'], batch['src_img'].to(device)
            src_mask = model.extract_cam(src_img, args.backbone)
            torch.save(src_mask.cpu(), os.path.join(f0_folder, '%s.pt'%(img_imname[0][:-4])))

            trg_img = torch.flip(src_img, dims=(3,))
            trg_mask = model.extract_cam(trg_img, args.backbone) 
            torch.save(trg_mask.cpu(), os.path.join(f1_folder, '%s.pt'%(img_imname[0][:-4])))
            # img = transform(mask)
            # img.save()
            # print(torch.max(src_mask), torch.max(trg_mask))
            
            # print(mask.size(), img_imname[0][:-4])
            




