"""Provides functions that manipulate boxes and points"""

import torch

from . import util


def center(box):
    r"""Calculates centers, (x, y), of box (N, 4)"""
    x_center = box[:, 0] + torch.div((box[:, 2] - box[:, 0]), 2, rounding_mode='trunc')
    y_center = box[:, 1] + torch.div((box[:, 3] - box[:, 1]), 2, rounding_mode='trunc')
    # print("x_center", x_center.size())
    return torch.stack((x_center, y_center)).t().to(box.device)


def receptive_fields(rfsz, jsz, feat_size):
    r"""Returns a set of receptive fields (N, 4)"""

    width = feat_size[3]
    height = feat_size[2]

    # print(width, height)

    feat_ids = torch.tensor(list(range(width))).repeat(1, height).t().repeat(1, 2).to(jsz.device)
    feat_ids[:, 0] = torch.tensor(list(range(height))).unsqueeze(1).repeat(1, width).view(-1)

    box = torch.zeros(feat_ids.size()[0], 4).to(jsz.device)
    # print(box.device, feat_ids.device, jsz.device)
    box[:, 0] = feat_ids[:, 1] * jsz - torch.div(rfsz, 2, rounding_mode='trunc')
    box[:, 1] = feat_ids[:, 0] * jsz - torch.div(rfsz, 2, rounding_mode='trunc')
    box[:, 2] = feat_ids[:, 1] * jsz + torch.div(rfsz, 2, rounding_mode='trunc')
    box[:, 3] = feat_ids[:, 0] * jsz + torch.div(rfsz, 2, rounding_mode='trunc')

    return box

def prune_margin(receptive_box, imsize, threshold=0):
    r"""Remove receptive fields on the margin of the image"""
    im_width = imsize[1]
    im_height = imsize[0]

    xmin = util.where(receptive_box[:, 0] >= 0 - threshold)
    ymin = util.where(receptive_box[:, 1] >= 0 - threshold)
    xmax = util.where(receptive_box[:, 2] < im_width + threshold)
    ymax = util.where(receptive_box[:, 3] < im_height + threshold)

    val1 = util.intersect1d(xmin, ymin)
    val2 = util.intersect1d(xmax, ymax)
    valid_ids = torch.sort(torch.unique(util.intersect1d(val1, val2)))[0]

    pruned_receptive_box = receptive_box[valid_ids]

    return pruned_receptive_box, valid_ids


def prune_bbox(receptive_box, bbox, threshold=0):
    r"""Remove receptive fields on the margin of the image"""
    xmin = util.where(receptive_box[:, 0] >= bbox[0] - threshold)
    ymin = util.where(receptive_box[:, 1] >= bbox[1] - threshold)
    xmax = util.where(receptive_box[:, 2] < bbox[2] + threshold)
    ymax = util.where(receptive_box[:, 3] < bbox[3] + threshold)

    val1 = util.intersect1d(xmin, ymin)
    val2 = util.intersect1d(xmax, ymax)
    valid_ids = torch.sort(torch.unique(util.intersect1d(val1, val2)))[0]

    pruned_receptive_box = receptive_box[valid_ids]

    return pruned_receptive_box, valid_ids

def mutual_nn_filter(correlation_matrix):
    r"""Mutual nearest neighbor filtering (Rocco et al. NeurIPS'18)"""
    corr_src_max = torch.max(correlation_matrix, dim=2, keepdim=True)[0]
    corr_trg_max = torch.max(correlation_matrix, dim=1, keepdim=True)[0]
    corr_src_max[corr_src_max == 0] = corr_src_max[corr_src_max == 0] + 1e-30
    corr_trg_max[corr_trg_max == 0] = corr_trg_max[corr_trg_max == 0] + 1e-30

    corr_src = correlation_matrix / (corr_src_max + 1e-30)
    corr_trg = correlation_matrix / (corr_trg_max + 1e-30)

    return correlation_matrix * (corr_src * corr_trg)

def predict_kps(src_box, trg_box, src_kps, n_pts, confidence_ts):
    r"""Transfer keypoints by nearest-neighbour assignment"""

    # TODO: """Mutual nearest neighbor filtering (Rocco et al. NeurIPS'18)"""
    # confidence_ts = mutual_nn_filter(confidence_ts) # refined correleation matrix

    prd_kps = []
    max_pts = 40
    # print(src_box.size())
    for ct, kpss, np in zip(confidence_ts, src_kps, n_pts):

        # print(ct.size(), kpss.size(), np)
        
        # 1. Prepare geometries & argmax target indices
        kp = kpss.narrow_copy(1, 0, np) # cut the real kpss
        _, trg_argmax_idx = torch.max(ct, dim=1)
        src_geomet = src_box[:, :2].unsqueeze(0).repeat(len(kp.t()), 1, 1)
        trg_geomet = trg_box[:, :2].unsqueeze(0).repeat(len(kp.t()), 1, 1)

        # 2. Retrieve neighbouring source boxes that cover source key-points
        src_nbr_onehot, n_neighbours = neighbours(src_box, kp)

        # 3. Get displacements from source neighbouring box centers to each key-point
        src_displacements = kp.t().unsqueeze(1).repeat(1, len(src_box), 1) - src_geomet
        src_displacements = src_displacements * src_nbr_onehot.unsqueeze(2).repeat(1, 1, 2).float()

        # 4. Transfer the neighbours based on given confidence tensor
        vector_summator = torch.zeros_like(src_geomet)
        src_idx = src_nbr_onehot.nonzero()
        trg_idx = trg_argmax_idx.index_select(dim=0, index=src_idx[:, 1])
        vector_summator[src_idx[:, 0], src_idx[:, 1]] = trg_geomet[src_idx[:, 0], trg_idx]
        vector_summator += src_displacements
        prd = (vector_summator.sum(dim=1) / n_neighbours.unsqueeze(1).repeat(1, 2).float()).t()

        # 5. Concatenate pad-points for batch
        pads = (-100*torch.ones((2, max_pts - np))).to(prd.device)
        prd = torch.cat([prd, pads], dim=1)
        prd_kps.append(prd)

    return torch.stack(prd_kps)


def neighbours(box, kps):
    r"""Returns boxes in one-hot format that covers given keypoints"""
    box_duplicate = box.unsqueeze(2).repeat(1, 1, len(kps.t())).transpose(0, 1)
    kps_duplicate = kps.unsqueeze(1).repeat(1, len(box), 1)
    
    # print(box_duplicate.is_cuda, kps_duplicate.is_cuda)

    xmin = kps_duplicate[0].ge(box_duplicate[0])
    ymin = kps_duplicate[1].ge(box_duplicate[1])
    xmax = kps_duplicate[0].le(box_duplicate[2])
    ymax = kps_duplicate[1].le(box_duplicate[3])

    nbr_onehot = torch.mul(torch.mul(xmin, ymin), torch.mul(xmax, ymax)).t()
    n_neighbours = nbr_onehot.sum(dim=1)

    return nbr_onehot, n_neighbours

def predict_test_kps(src_box, trg_box, src_kps, confidence_ts):
    r"""Transfer keypoints by nearest-neighbour assignment"""

    # 1. Prepare geometries & argmax target indices
    _, trg_argmax_idx = torch.max(confidence_ts, dim=1)
    src_geomet = src_box[:, :2].unsqueeze(0).repeat(len(src_kps.t()), 1, 1)
    trg_geomet = trg_box[:, :2].unsqueeze(0).repeat(len(src_kps.t()), 1, 1)

    # 2. Retrieve neighbouring source boxes that cover source key-points
    src_nbr_onehot, n_neighbours = neighbours(src_box, src_kps)

    # 3. Get displacements from source neighbouring box centers to each key-point
    src_displacements = src_kps.t().unsqueeze(1).repeat(1, len(src_box), 1) - src_geomet
    src_displacements = src_displacements * src_nbr_onehot.unsqueeze(2).repeat(1, 1, 2).float()

    # 4. Transfer the neighbours based on given confidence tensor
    vector_summator = torch.zeros_like(src_geomet)
    src_idx = src_nbr_onehot.nonzero()
    trg_idx = trg_argmax_idx.index_select(dim=0, index=src_idx[:, 1])
    vector_summator[src_idx[:, 0], src_idx[:, 1]] = trg_geomet[src_idx[:, 0], trg_idx]
    vector_summator += src_displacements
    pred = (vector_summator.sum(dim=1) / n_neighbours.unsqueeze(1).repeat(1, 2).float())

    return pred.t()

def gaussian2d(side=7):
    r"""Returns 2-dimensional gaussian filter"""
    dim = [side, side]

    siz = torch.LongTensor(dim)
    sig_sq = (siz.float()/2/2.354).pow(2)
    siz2 = (siz-1)/2

    x_axis = torch.arange(-siz2[0], siz2[0] + 1).unsqueeze(0).expand(dim).float()
    y_axis = torch.arange(-siz2[1], siz2[1] + 1).unsqueeze(1).expand(dim).float()

    gaussian = torch.exp(-(x_axis.pow(2)/2/sig_sq[0] + y_axis.pow(2)/2/sig_sq[1]))
    gaussian = gaussian / gaussian.sum()

    return gaussian
