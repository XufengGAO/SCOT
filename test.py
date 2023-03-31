import torch
import torch.nn as nn
from data import download
from model.base.geometry import Geometry

from torchvision.utils import draw_keypoints
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F
from model.base.geometry import Geometry
import random
from model import scot_CAM, util, geometry
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
import os
from torchvision import transforms
import matplotlib.patches as patches
# plt.ion()

benchmark = "pfpascal"
backbone = "resnet101"
datapath="./Datasets_SCOT/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cam=""
seed=0
thres="auto"
split="test"

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True




hyperpixel = list(range(34))
model = scot_CAM.SCOT_CAM(backbone, hyperpixel, benchmark, device, cam, use_xavier=True)
dset = download.load_dataset(benchmark, datapath, thres, device, split, cam, img_side=(256,256))
dataloader = DataLoader(dset, batch_size=2, num_workers=0)
print('loader_size', len(dataloader))

def get_concat_h(im1, im2):
    max_height = max(im1.height, im2.height)
    dst = Image.new('RGB', (im1.width + im2.width, max_height), color="white")
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def draw_keypoints_on_image(src_img, trg_img, src_kps, trg_kps, n_pts, color='red',
                            radius=2):
    
    r"""Draw keypoints on image
    
    Args:
        src_img, trg_img: The original PIL.Image object
        src_kps, trg_kps: a tensor with shape [2, n_pts]

    """

    # 1. concatenate images horizontally
    con_img = get_concat_h(src_img, trg_img)

    # 2. draw keypoints and connected lines
    draw = ImageDraw.Draw(con_img)
    for idx in range(n_pts):
        draw.ellipse([(src_kps[0,idx] - radius, src_kps[1,idx] - radius),
                    (src_kps[0,idx] + radius, src_kps[1,idx] + radius)],
                    outline=color, fill=color)
        draw.ellipse([(trg_kps[0,idx] - radius + con_img.width//2, trg_kps[1,idx] - radius),
            (trg_kps[0,idx] + radius + con_img.width//2, trg_kps[1,idx] + radius)],
            outline=color, fill=color)
        draw.line((src_kps[0,idx], src_kps[1,idx], 
                   trg_kps[0,idx]+ con_img.width//2, trg_kps[1,idx]), joint="curve", fill=color, width=5)
        
        

    return con_img



imside = (256,256) # HxW

trns = T.ToPILImage()
for idx, sample in enumerate(dataloader):
    print(sample['src_imname'], sample['trg_imname'])

    # 1. original PIL image
    src_img = dataloader.dataset.get_image(sample['src_imname'][0])
    trg_img = dataloader.dataset.get_image(sample['trg_imname'][0])
    src_img = src_img.resize(imside) # (width, height). 
    trg_img =  trg_img.resize(imside)

    n_pts = sample['n_pts'][0]

    src_ratio = sample['src_ratio'][0].flip(dims=(0,)).view(2,-1)
    trg_ratio = sample['trg_ratio'][0].flip(dims=(0,)).view(2,-1)

    # print(src_ratio.size(), sample['src_kps'][0].size())

    src_ratio, trg_ratio = 1.0, 1.0

    # rescale to original points
    src_kps = (sample['src_kps'][0][:,:n_pts.item()] / src_ratio).numpy()
    trg_kps = (sample['trg_kps'][0][:,:n_pts.item()] / trg_ratio).numpy()
    # print(src_kps)


    src_bbox = sample['src_bbox'][0].numpy()
    trg_bbox = sample['trg_bbox'][0].numpy()
    src_bbox_start = (src_bbox[0], src_bbox[1])
    src_bbox_w, src_bbox_h = src_bbox[2] - src_bbox[0], src_bbox[3] - src_bbox[1]
    trg_bbox_start = (trg_bbox[0] + imside[1], trg_bbox[1])
    trg_bbox_w, trg_bbox_h = trg_bbox[2] - trg_bbox[0], trg_bbox[3] - trg_bbox[1]

    # con_img =  draw_keypoints_on_image(src_img, trg_img, src_kps, trg_kps, n_pts, color='red',radius=4)
    # con_img.show()
    # con_img.close()

    src_rect = patches.Rectangle(src_bbox_start, src_bbox_w, src_bbox_h, linewidth=2, edgecolor='b', facecolor='none')
    trg_rect = patches.Rectangle(trg_bbox_start, trg_bbox_w, trg_bbox_h, linewidth=2, edgecolor='b', facecolor='none')

    con_img = get_concat_h(src_img, trg_img)
    con_img = np.array(con_img)
    print(con_img.shape)

    fig, ax = plt.subplots()
    ax.imshow(con_img)

    for idx in range(n_pts):
        ax.plot(src_kps[0,idx], src_kps[1,idx], marker='o', color="red")
        ax.plot(trg_kps[0,idx] + imside[1], trg_kps[1,idx], marker='o', color="red")
        ax.plot([src_kps[0,idx], trg_kps[0,idx] + imside[1]], [src_kps[1,idx], trg_kps[1,idx]], color="red", linewidth=2)

    # Add the patch to the Axes
    ax.add_patch(src_rect)
    ax.add_patch(trg_rect)

    # plt.show()
    # plt.close("all")

    break


def draw_matches_on_image(src_img, trg_img, batch, origin=True, pck_ids=None):
    r"""Draw keypoints on image
    
    Args:
        src_img, trg_img: The original PIL.Image object
        src_kps, trg_kps: a tensor with shape [2, n_pts]
        

    """
    idx = 0
    if batch['flip'][idx].item() == 1:
        src_img = src_img.transpose(Image.FLIP_LEFT_RIGHT)
        trg_img = trg_img.transpose(Image.FLIP_LEFT_RIGHT)
    n_pts = batch['n_pts'][idx]

    if origin:
        src_ratio = batch['src_ratio'][idx].flip(dims=(0,)).view(2,-1) # wxh
        trg_ratio = batch['trg_ratio'][idx].flip(dims=(0,)).view(2,-1)
    else:
        src_ratio = torch.ones((2,1)) 
        trg_ratio = torch.ones((2,1))

    # Rescale image
    src_kps = (batch['src_kps'][idx][:,:n_pts.item()] / src_ratio).numpy()
    trg_kps = (batch['trg_kps'][idx][:,:n_pts.item()] / trg_ratio).numpy()
    
    # 
    src_bbox = batch['src_bbox'][idx].numpy()
    trg_bbox = batch['trg_bbox'][idx].numpy()
    src_bbox_start = (src_bbox[0]/src_ratio.numpy()[0], src_bbox[1]/src_ratio.numpy()[1])
    src_bbox_w, src_bbox_h = (src_bbox[2] - src_bbox[0])/src_ratio.numpy()[0], (src_bbox[3] - src_bbox[1])/src_ratio.numpy()[1]

    trg_bbox_start = ((trg_bbox[0]/trg_ratio.numpy()[0] + src_img.width), trg_bbox[1]/trg_ratio.numpy()[1])
    trg_bbox_w, trg_bbox_h = (trg_bbox[2] - trg_bbox[0])/trg_ratio.numpy()[0], (trg_bbox[3] - trg_bbox[1])/trg_ratio.numpy()[1]

    src_rect = patches.Rectangle(src_bbox_start, src_bbox_w, src_bbox_h, linewidth=2, edgecolor='b', facecolor='none')
    trg_rect = patches.Rectangle(trg_bbox_start, trg_bbox_w, trg_bbox_h, linewidth=2, edgecolor='b', facecolor='none')

    con_img = get_concat_h(src_img, trg_img)
    con_img = np.array(con_img)
    fig, ax = plt.subplots()
    ax.imshow(con_img)

    # plot lines and keypoints
    colors = ['red', 'green']
    if pck_ids is None:
        pck_ids = torch.ones(n_pts, dtype=torch.uint8)

    for idx in range(n_pts):
        ax.plot(src_kps[0,idx], src_kps[1,idx], marker='o', color=colors[pck_ids[idx]])
        ax.plot(trg_kps[0,idx] + src_img.width, trg_kps[1,idx], marker='o', color=colors[pck_ids[idx]])
        ax.plot([src_kps[0,idx], trg_kps[0,idx] + src_img.width], [src_kps[1,idx], trg_kps[1,idx]], color=colors[pck_ids[idx]], linewidth=2)

    ax.add_patch(src_rect)
    ax.add_patch(trg_rect)

    plt.savefig('')

    return fig

