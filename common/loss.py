r"""Two different strategies of weak/strong supervisions"""
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import numpy as np
from model.objective import Objective
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LossStrategy(ABC):
    r"""Different strategies for methods:"""
    @abstractmethod
    def get_image_pair(self, batch, *args):
        pass

    @abstractmethod
    def get_correlation(self, correlation_matrix):
        pass

    @abstractmethod
    def compute_loss(self, correlation_matrix, *args):
        pass


class StrongCELoss(LossStrategy):
    def get_image_pair(self, batch, *args):
        r"""Returns (semantically related) pairs for strongly-supervised training"""
        return batch['src_img'].to(device), batch['trg_img'].to(device)

    def get_correlation(self, correlation_matrix):
        r"""Returns correlation matrices of 'ALL PAIRS' in a batch"""
        return correlation_matrix.detach().clone()

    def compute_loss(self, correlation_matrix, eval_result, pckthres, n_pts):
        r"""Strongly-supervised matching loss (L_{match})"""

        loss_cre = Objective.weighted_cross_entropy(correlation_matrix, eval_result['easy_match'], eval_result['hard_match'], pckthres, n_pts)
        # loss_sel = Objective.layer_selection_loss(layer_sel)
        # loss_net = loss_cre + loss_sel

        return loss_cre
    
class WeakLoss(LossStrategy):
    def get_image_pair(self, batch, *args):
        r"""Returns (semantically related) pairs for strongly-supervised training"""
        return batch['src_img'].to(device), batch['trg_img'].to(device)

    def get_correlation(self, correlation_matrix):
        r"""Returns correlation matrices of 'ALL PAIRS' in a batch"""
        return correlation_matrix.detach().clone()

    def compute_loss(self, cross_sim, src_sim, trg_sim, src_feats, trg_feats, weak_norm='l1', temp=1.0):
        r"""Strongly-supervised matching loss (L_{match})"""
        
        # 1. cross-entropy loss for self-correleation maps
        # print(src_sim.size(), trg_sim.size(), src_feats.size(), trg_feats.size(), cross_sim.size())
        discSelf_loss = Objective.infor_entropy(src_sim, weak_norm) + Objective.infor_entropy(trg_sim, weak_norm)
        discCross_loss = Objective.infor_entropy(cross_sim, weak_norm)
        # disc_loss = torch.zeros(1).to(cross_sim.device)
        
        src_feats = src_feats / (torch.norm(src_feats, p=2, dim=2).unsqueeze(-1)+ 1e-10) # normalized features
        trg_feats = trg_feats / (torch.norm(trg_feats, p=2, dim=2).unsqueeze(-1)+ 1e-10)

        # 2. matching loss for features
        src2trg_dist = torch.bmm(src_feats.transpose(1,2), F.softmax(cross_sim/temp, dim=1)) - trg_feats.transpose(1,2)
        trg2src_dist = torch.bmm(trg_feats.transpose(1,2), F.softmax((cross_sim.transpose(1,2))/temp, dim=1)) - src_feats.transpose(1,2)
        #src2trg_dist = src2trg_dist / (torch.norm(src2trg_dist, p=2, dim=1, keepdim=True)+ 1e-10)
        #trg2src_dist = trg2src_dist / (torch.norm(trg2src_dist, p=2, dim=1, keepdim=True)+ 1e-10)

        match_loss = 0.5 * (src2trg_dist.norm(dim=(1,2)).mean() + trg2src_dist.norm(dim=(1,2)).mean())
        # match_loss = torch.zeros(1).to(cross_sim.device)

        return discSelf_loss, discCross_loss, match_loss

class StrongFlowLoss(LossStrategy):
    def get_image_pair(self, batch, *args):
        r"""Returns (semantically related) pairs for strongly-supervised training"""
        return batch['src_img'], batch['trg_img']

    def get_correlation(self, correlation_matrix):
        r"""Returns correlation matrices of 'ALL PAIRS' in a batch"""
        return correlation_matrix.detach().clone()

    def compute_loss(self, correlation_matrix, flow_gt, feat_size):
        r"""Strongly-supervised matching loss (L_{match})"""
        
        B = correlation_matrix.size()[0]
        grid_x, grid_y = soft_argmax(correlation_matrix.view(B, -1, feat_size, feat_size), feature_size=feat_size)

        pred_flow = torch.cat((grid_x, grid_y), dim=1)
        pred_flow = unnormalise_and_convert_mapping_to_flow(pred_flow)

        loss_flow = EPE(pred_flow, flow_gt)
        return loss_flow
    
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

def soft_argmax(corr, beta=0.02, feature_size=64):
    r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
    x_normal = nn.Parameter(torch.tensor(np.linspace(-1,1,feature_size), dtype=torch.float, requires_grad=False)).to(device)
    y_normal = nn.Parameter(torch.tensor(np.linspace(-1,1,feature_size), dtype=torch.float, requires_grad=False)).to(device)

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


class WeakLoss2(LossStrategy):
    def get_image_pair(self, batch, *args):
        r"""Forms positive/negative image paris for weakly-supervised training"""
        training = args[0]
        self.bsz = len(batch['src_img'])

        if training:
            shifted_idx = np.roll(np.arange(self.bsz), -1)
            trg_img_neg = batch['trg_img'][shifted_idx].clone()
            trg_cls_neg = batch['category_id'][shifted_idx].clone()
            neg_subidx = (batch['category_id'] - trg_cls_neg) != 0

            src_img = torch.cat([batch['src_img'], batch['src_img'][neg_subidx]], dim=0)
            trg_img = torch.cat([batch['trg_img'], trg_img_neg[neg_subidx]], dim=0)
            self.num_negatives = neg_subidx.sum()
        else:
            src_img, trg_img = batch['src_img'], batch['trg_img']
            self.num_negatives = 0

        return src_img, trg_img

    def get_correlation(self, correlation_matrix):
        r"""Returns correlation matrices of 'POSITIVE PAIRS' in a batch"""
        return correlation_matrix[:self.bsz].detach().clone()

    def compute_loss(self, correlation_matrix, *args):
        r"""Weakly-supervised matching loss (L_{match})"""
        layer_sel = args[1]
        loss_pos = Objective.information_entropy(correlation_matrix[:self.bsz])
        loss_neg = Objective.information_entropy(correlation_matrix[self.bsz:]) if self.num_negatives > 0 else 1.0
        loss_sel = Objective.layer_selection_loss(layer_sel)
        loss_net = (loss_pos / loss_neg) + loss_sel

        return loss_net
    
    

