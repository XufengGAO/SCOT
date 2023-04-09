r"""Some helper functions"""
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

def fix_randseed(seed):
    r"""Fixes random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mean(x):
    r"""Computes average of a list"""
    return sum(x) / len(x) if len(x) > 0 else 0.0


def where(predicate):
    r"""Predicate must be a condition on nd-tensor"""
    matching_indices = predicate.nonzero()
    if len(matching_indices) != 0:
        matching_indices = matching_indices.t().squeeze(0)
    return matching_indices


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def convert_weight_map(weight):

    num_weight = weight.numel()
    pad_width = 20 if num_weight == 17 else 35
    pad_weight = torch.zeros(pad_width, dtype=weight.dtype)
    pad_weight[:num_weight] = weight
    pad_weight[num_weight:] = -1000

    pad_weight = pad_weight.sigmoid().view(-1, 5)

    return pad_weight

def draw_class_pck(sel_buffer, class_pck_path, epoch=0, step=0):

    mean_sel_buffer = {}
    for (key, value) in sel_buffer.items():
        # print(key, "%.3f" % list_mean(value))
        mean_sel_buffer[key] = mean(value)

    names = list(mean_sel_buffer.keys())
    values = list(mean_sel_buffer.values())

    fig, axs = plt.subplots(1, 1, figsize=(8, 3), sharey=True)
    pps = axs.bar(names, values)

    plt.setp(axs.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    axs.set_ylabel('Avg pck')

    for p in pps:
        height = p.get_height()
        axs.annotate('%.2f'%(height),
                        xy=(p.get_x()+p.get_width()/2, height),
                        xytext=(0,3),
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=45)
    axs.set_title("Per-class pck, e={}, s={}".format(epoch, step))
    fig.tight_layout()

    class_pth = os.path.join(class_pck_path, "e{}_s{}.png".format(epoch, step))

    fig.savefig(class_pth)
    plt.close('all')
    
    return class_pth

def draw_weight_map(weight, epoch, step, weight_map_path):

    num_weight = weight.numel()
    pad_width = 20 if num_weight == 17 else 35
    pad_weight = torch.zeros(pad_width, dtype=weight.dtype)
    pad_weight[:num_weight] = weight
    pad_weight[num_weight:] = -1000

    pad_weight = pad_weight.sigmoid().view(-1, 5)

    y, x = pad_weight.size()[0], pad_weight.size()[1]
    fig, ax = plt.subplots()
    im = ax.imshow(pad_weight)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(x))
    ax.set_yticks(np.arange(y))

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(y):
        for j in range(x):
            text = ax.text(j, i, "%.3f"%pad_weight[i, j].item(),
                        ha="center", va="center", color="w")

    ax.set_title("Sigmoid weight, epoch={}, step={}".format(epoch, step))
    fig.tight_layout()

    weight_pth = os.path.join(weight_map_path, "e{}_s{}.png".format(epoch, step))

    fig.savefig(weight_pth)
    plt.close('all')
    
    return weight_pth

def get_concat_h(im1, im2):
    max_height = max(im1.height, im2.height)
    dst = Image.new('RGB', (im1.width + im2.width, max_height), color="white")
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def draw_matches_on_image(epoch, step, match_idx, match_pck, src_img, trg_img, batch, pred_trg_kps=None, origin=True, color_ids=None, draw_match_path=None):
    r"""Draw keypoints on image
    
    Args:
        match_idx: sample id in one batch
        src_img, trg_img: The original PIL.Image object

    """
    
    # 1. Check flip
    if batch['flip'][match_idx].item() == 1:
        src_img = src_img.transpose(Image.FLIP_LEFT_RIGHT)
        trg_img = trg_img.transpose(Image.FLIP_LEFT_RIGHT)
    n_pts = batch['n_pts'][match_idx]

    # 2. Check image rescale
    if origin:
        src_ratio = batch['src_intratio'][match_idx].flip(dims=(0,)).view(2,-1).cpu().numpy() # wxh
        trg_ratio = batch['trg_intratio'][match_idx].flip(dims=(0,)).view(2,-1).cpu().numpy()
    else:
        src_ratio = torch.ones((2,1)).cpu().numpy() 
        trg_ratio = torch.ones((2,1)).cpu().numpy()

    # 3. Check kps
    src_kps = (batch['src_kps'][match_idx][:,:n_pts.item()].cpu().numpy() / src_ratio)

    if pred_trg_kps is not None:
        trg_kps = (pred_trg_kps[:,:n_pts.item()].cpu().numpy() / trg_ratio)
    else:
        trg_kps = (batch['trg_kps'][match_idx][:,:n_pts.item()].cpu().numpy() / trg_ratio).cpu().numpy()
    
    # 4. Check bounding box
    src_bbox = batch['src_bbox'][match_idx].cpu().numpy()
    trg_bbox = batch['trg_bbox'][match_idx].cpu().numpy()

    src_bbox_start = (src_bbox[0]/src_ratio[0], src_bbox[1]/src_ratio[1])
    src_bbox_w, src_bbox_h = (src_bbox[2] - src_bbox[0])/src_ratio[0], (src_bbox[3] - src_bbox[1])/src_ratio[1]

    trg_bbox_start = ((trg_bbox[0]/trg_ratio[0] + src_img.width), trg_bbox[1]/trg_ratio[1])
    trg_bbox_w, trg_bbox_h = (trg_bbox[2] - trg_bbox[0])/trg_ratio[0], (trg_bbox[3] - trg_bbox[1])/trg_ratio[1]

    src_rect = patches.Rectangle(src_bbox_start, src_bbox_w, src_bbox_h, linewidth=2, edgecolor='b', facecolor='none')
    trg_rect = patches.Rectangle(trg_bbox_start, trg_bbox_w, trg_bbox_h, linewidth=2, edgecolor='b', facecolor='none')

    # 5. Concatenate images horinzontally
    con_img = get_concat_h(src_img, trg_img)
    con_img = np.array(con_img)

    # 6. Draw the matches
    fig, ax = plt.subplots()
    ax.imshow(con_img)

    colors = ['red', 'green']
    if color_ids is None:
        color_ids = torch.ones(n_pts, dtype=torch.uint8)

    for pt_idx in range(n_pts):
        ax.plot(src_kps[0,pt_idx], src_kps[1,pt_idx], marker='o', color=colors[color_ids[pt_idx]])
        ax.plot(trg_kps[0,pt_idx] + src_img.width, trg_kps[1,pt_idx], marker='o', color=colors[color_ids[pt_idx]])
        ax.plot([src_kps[0,pt_idx], trg_kps[0,pt_idx] + src_img.width], [src_kps[1,pt_idx], trg_kps[1,pt_idx]], color=colors[color_ids[pt_idx]], linewidth=2)

    ax.add_patch(src_rect)
    ax.add_patch(trg_rect)

    img_name = "{}-{}.png".format(batch["src_imname"][match_idx][:-4], batch["src_imname"][match_idx][:-4])
    ax.set_title("Pair=%s, epoch=%d, step=%d, pck=%.3f" % (img_name[:-4], epoch, step, match_pck))
    fig.tight_layout()

    match_pth = os.path.join(draw_match_path, img_name)

    fig.savefig(match_pth)
    plt.close('all')

    return match_pth

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']