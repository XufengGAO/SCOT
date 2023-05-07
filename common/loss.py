r"""Two different strategies of weak/strong supervisions"""
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import numpy as np
from model.objective import Objective
import torch.nn.functional as F
from .norm import unit_gaussian_normalize, l1normalize, linearnormalize
import re

class StrongCrossEntropyLoss(nn.Module):
    r"""Strongly-supervised cross entropy loss"""
    def __init__(self, alpha=0.1) -> None:
        super(StrongCrossEntropyLoss, self).__init__()
        assert alpha > 0.0, "negative alpha is not allowed"
        self.softmax = torch.nn.Softmax(dim=1)
        self.alpha = alpha
        self.eps = 1e-30

    def forward(self, x: torch.Tensor, easy_match, hard_match, pckthres, n_pts) -> torch.Tensor:
        loss_buf = x.new_zeros(x.size(0))

        # normalize each row of coefficient, de-mean and unit-std
        x = unit_gaussian_normalize(x)

        for idx, (ct, thres, npt) in enumerate(zip(x, pckthres, n_pts)):

            # Hard (incorrect) match
            if len(hard_match['src'][idx]) > 0:
                cross_ent = self.cross_entropy(ct, hard_match['src'][idx], hard_match['trg'][idx])
                loss_buf[idx] += cross_ent.sum()

            # Easy (correct) match
            if len(easy_match['src'][idx]) > 0:
                cross_ent = self.cross_entropy(ct, easy_match['src'][idx], easy_match['trg'][idx])
                smooth_weight = (easy_match['dist'][idx] / (thres * self.alpha)).pow(2)
                loss_buf[idx] += (smooth_weight * cross_ent).sum()

            loss_buf[idx] /= npt

        return loss_buf.mean()
    
    def cross_entropy(self, correlation_matrix, src_match, trg_match):
        r"""Cross-entropy between predicted pdf and ground-truth pdf (one-hot vector)"""        
        pdf = self.softmax(correlation_matrix.index_select(0, src_match))
        # print("pdf", pdf.size(), trg_match)
        prob = pdf[range(len(trg_match)), trg_match.long()]
        cross_ent = -torch.log(prob + self.eps)

        return cross_ent
    
class WeakDiscMatchLoss(nn.Module):
    r"""Weakly-supervised discriminative and maching loss"""
    def __init__(self, temp=1.0,  match_norm_type='l1', weak_lambda=None) -> None:
        super(WeakDiscMatchLoss, self).__init__()
        self.softmax = torch.nn.Softmax(dim=1)
        self.eps = 1e-30
        self.temp = temp
        self.match_norm_type = match_norm_type
        weak_lambda = list(map(float, re.findall(r"[-+]?(?:\d*\.*\d+)", weak_lambda)))
        self.weak_lambda = [i>0.0 for i in weak_lambda]

    def forward(self, x_cross: torch.Tensor, x_src: torch.Tensor, x_trg: torch.Tensor, src_feats: torch.Tensor, trg_feats: torch.Tensor) -> torch.Tensor:
        
        # match_loss = torch.zeros(1).to(x_cross.device)
        if self.weak_lambda[0]:
            discSelf_loss = 0.5*(self.information_entropy(x_src, self.match_norm_type) + self.information_entropy(x_trg, self.match_norm_type))
        else:
            discSelf_loss = torch.zeros(1).cuda()

        if self.weak_lambda[1]:
            discCross_loss = self.information_entropy(x_cross, self.match_norm_type)
        else:
            discCross_loss = torch.zeros(1).cuda()

        if self.weak_lambda[2]:
            match_loss = self.information_match(x_cross, src_feats, trg_feats)
        else:
            match_loss = torch.zeros(1).cuda()
                    
        task_loss = torch.stack([discSelf_loss.mean(), discCross_loss.mean(), match_loss.mean()])
        return task_loss


    def information_entropy(self, correlation_matrix, norm_type='l1'):
        r"""Computes information entropy of all candidate matches"""
        #correlation_matrix = Correlation.mutual_nn_filter(correlation_matrix)

        norm = {'l1':l1normalize, 'linear':linearnormalize}
        src_pdf = norm[norm_type](correlation_matrix)
        trg_pdf = norm[norm_type](correlation_matrix.transpose(1,2))

        src_pdf[src_pdf <= self.eps] = self.eps
        trg_pdf[trg_pdf <= self.eps] = self.eps

        src_ent = (-(src_pdf * torch.log2(src_pdf)).sum(dim=2))
        trg_ent = (-(trg_pdf * torch.log2(trg_pdf)).sum(dim=2))

        score_net = ((src_ent + trg_ent).mean(dim=1) / 2)
        del src_ent, trg_ent, src_pdf, trg_pdf, correlation_matrix
        
        return score_net
    
    def information_match(self, x_cross: torch.Tensor, src_feats: torch.Tensor, trg_feats: torch.Tensor):
        src_feats = src_feats / (torch.norm(src_feats, p=2, dim=2, keepdim=True)+ self.eps) # normalized features
        trg_feats = trg_feats / (torch.norm(trg_feats, p=2, dim=2, keepdim=True)+ self.eps)

        # 2. matching loss for features
        src2trg_dist = torch.bmm(src_feats.transpose(1,2), self.softmax(x_cross/self.temp)) - trg_feats.transpose(1,2)
        trg2src_dist = torch.bmm(trg_feats.transpose(1,2), self.softmax(x_cross.transpose(1,2)/self.temp)) - src_feats.transpose(1,2)
        #src2trg_dist = src2trg_dist / (torch.norm(src2trg_dist, p=2, dim=1, keepdim=True)+ 1e-10)
        #trg2src_dist = trg2src_dist / (torch.norm(trg2src_dist, p=2, dim=1, keepdim=True)+ 1e-10)

        match_loss = 0.5 * (src2trg_dist.norm(dim=(1)).mean(dim=1) + trg2src_dist.norm(dim=(1)).mean(dim=1))
        del src_feats, trg_feats, src2trg_dist, trg2src_dist

        return match_loss


class StrongFlowLoss(nn.Module):
    r"""Strongly-supervised flow loss"""
    def __init__(self) -> None:
        super(StrongFlowLoss, self).__init__()
        self.eps = 1e-30

    def forward(self, x: torch.Tensor, flow_gt, feat_size):
        r"""Strongly-supervised matching loss (L_{match})"""
        
        B = x.size()[0]
        grid_x, grid_y = soft_argmax(x.view(B, -1, feat_size, feat_size), feature_size=feat_size)

        pred_flow = torch.cat((grid_x, grid_y), dim=1)
        pred_flow = unnormalise_and_convert_mapping_to_flow(pred_flow)

        loss_flow = EPE(pred_flow, flow_gt)
        return loss_flow
    
def soft_argmax(self, corr, beta=0.02, feature_size=64):
    r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
    x_normal = nn.Parameter(torch.tensor(np.linspace(-1,1,feature_size), dtype=torch.float, requires_grad=False)).cuda()
    y_normal = nn.Parameter(torch.tensor(np.linspace(-1,1,feature_size), dtype=torch.float, requires_grad=False)).cuda()

    b,_,h,w = corr.size()
    
    corr = softmax_with_temperature(corr, beta=beta, d=1)
    corr = corr.view(-1,h,w,h,w) # (target hxw) x (source hxw)

    grid_x = corr.sum(dim=1, keepdim=False) # marginalize to x-coord.
    x_normal = x_normal.expand(b,w)
    x_normal = x_normal.view(b,w,1,1)
    grid_x = (grid_x*x_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
    
    grid_y = corr.sum(dim=2, keepdim=False) # marginalize to y-coord.
    y_normal = y_normal.expand(b,h)
    y_normal = y_normal.view(b,h,1,1)
    grid_y = (grid_y*y_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
    return grid_x, grid_y

def EPE(input_flow, target_flow, sparse=True, mean=True, sum=False):

    EPE_map = torch.norm(target_flow-input_flow, 2, 1)
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)

        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    elif sum:
        return EPE_map.sum()
    else:
        return EPE_map.sum()/torch.sum(~mask)

def unnormalise_and_convert_mapping_to_flow(map):
    # here map is normalised to -1;1
    # we put it back to 0,W-1, then convert it to flow
    B, C, H, W = map.size()
    mapping = torch.zeros_like(map)
    # mesh grid
    mapping[:,0,:,:] = (map[:, 0, :, :].float().clone() + 1) * (W - 1) / 2.0 # unormalise
    mapping[:,1,:,:] = (map[:, 1, :, :].float().clone() + 1) * (H - 1) / 2.0 # unormalise

    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if mapping.is_cuda:
        grid = grid.cuda()
    flow = mapping - grid
    return flow

def softmax_with_temperature(x, beta, d = 1):
    r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
    M, _ = x.max(dim=d, keepdim=True)
    x = x - M # subtract maximum value for stability
    exp_x = torch.exp(x/beta)
    exp_x_sum = exp_x.sum(dim=d, keepdim=True)
    return exp_x / exp_x_sum




# class WeakLoss2(LossStrategy):
#     def get_image_pair(self, batch, *args):
#         r"""Forms positive/negative image paris for weakly-supervised training"""
#         training = args[0]
#         self.bsz = len(batch['src_img'])

#         if training:
#             shifted_idx = np.roll(np.arange(self.bsz), -1)
#             trg_img_neg = batch['trg_img'][shifted_idx].clone()
#             trg_cls_neg = batch['category_id'][shifted_idx].clone()
#             neg_subidx = (batch['category_id'] - trg_cls_neg) != 0

#             src_img = torch.cat([batch['src_img'], batch['src_img'][neg_subidx]], dim=0)
#             trg_img = torch.cat([batch['trg_img'], trg_img_neg[neg_subidx]], dim=0)
#             self.num_negatives = neg_subidx.sum()
#         else:
#             src_img, trg_img = batch['src_img'], batch['trg_img']
#             self.num_negatives = 0

#         return src_img, trg_img

#     def get_correlation(self, correlation_matrix):
#         r"""Returns correlation matrices of 'POSITIVE PAIRS' in a batch"""
#         return correlation_matrix[:self.bsz].detach().clone()

#     def compute_loss(self, correlation_matrix, *args):
#         r"""Weakly-supervised matching loss (L_{match})"""
#         layer_sel = args[1]
#         loss_pos = Objective.information_entropy(correlation_matrix[:self.bsz])
#         loss_neg = Objective.information_entropy(correlation_matrix[self.bsz:]) if self.num_negatives > 0 else 1.0
#         loss_sel = Objective.layer_selection_loss(layer_sel)
#         loss_net = (loss_pos / loss_neg) + loss_sel

#         return loss_net
    
    

