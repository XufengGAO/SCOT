"""For quantitative evaluation of DHPF"""
from skimage import draw
import numpy as np
import torch

from . import utils


class Evaluator:
    r"""Computes evaluation metrics of PCK, LT-ACC, IoU"""
    @classmethod
    def initialize(cls, alpha=0.1):

        cls.eval_func = cls.eval_kps_transfer
        cls.alpha = alpha

    @classmethod
    def evaluate(cls, prd_kps, batch, supervision):
        r"""Compute evaluation metric"""
        return cls.eval_func(prd_kps, batch, supervision)

    @classmethod
    def eval_kps_transfer(cls, prd_kps, batch, supervision):
        r"""Compute percentage of correct key-points (PCK) based on prediction"""

        easy_match = {'src': [], 'trg': [], 'dist': []}
        hard_match = {'src': [], 'trg': []}

        pck = []
        pck_ids = torch.zeros((prd_kps.size()[0], prd_kps.size()[-1]), dtype=torch.uint8) # Bx40, default incorrect points
        for idx, (pk, tk, thres, npt) in enumerate(zip(prd_kps, batch['trg_kps'], batch['pckthres'], batch['n_pts'])):
            correct_dist, correct_ids, incorrect_ids, ncorrt = cls.classify_prd(pk[:, :npt], tk[:, :npt], thres)
            # print(correct_ids, incorrect_ids)
            # print("correct", pk[:,correct_ids], tk[:,correct_ids])
            # print("incorrect", pk[:,incorrect_ids], tk[:,incorrect_ids])
            # pck_ids.append({'correct_ids':correct_ids, 'incorrect_ids':incorrect_ids})
            pck_ids[idx, correct_ids] = 1
            # Collect easy and hard match feature index & store pck to buffer
            if supervision == "strong_ce":
                easy_match['dist'].append(correct_dist)
                # for each keypoint, we find its nearest neighbour of center of receptive field
                # then kpidx is the id of hyperpixel
                easy_match['src'].append(batch['src_kpidx'][idx][:npt][correct_ids])
                easy_match['trg'].append(batch['trg_kpidx'][idx][:npt][correct_ids])
                hard_match['src'].append(batch['src_kpidx'][idx][:npt][incorrect_ids])
                hard_match['trg'].append(batch['trg_kpidx'][idx][:npt][incorrect_ids])
            pck.append(int(ncorrt)/int(npt))
            # print(int(ncorrt)/int(npt))
            del correct_dist, correct_ids, incorrect_ids, ncorrt
        
        eval_result = {'easy_match': easy_match,
                       'hard_match': hard_match,
                       'pck': pck, 
                       'pck_ids': pck_ids}
        del pck_ids
        
        return eval_result

    @classmethod
    def classify_prd(cls, prd_kps, trg_kps, pckthres):
        r"""Compute the number of correctly transferred key-points"""
        l2dist = (prd_kps - trg_kps).pow(2).sum(dim=0).pow(0.5)
        thres = pckthres.expand_as(l2dist).float() * cls.alpha
        correct_pts = torch.le(l2dist, thres)

        # print("l2", l2dist)
        # print("thres", thres)

        correct_ids = utils.where(correct_pts == 1)
        incorrect_ids = utils.where(correct_pts == 0)
        correct_dist = l2dist[correct_pts]

        del l2dist, thres

        return correct_dist, correct_ids, incorrect_ids, int(torch.sum(correct_pts))

