r"""SPair-71k dataset"""
import json
import glob
import os

import numpy as np
import torch

from .dataset import CorrespondenceDataset
from PIL import Image


class SPairDataset(CorrespondenceDataset):
    r"""Inherits CorrespondenceDataset"""
    def __init__(self, benchmark, datapath, thres, device, split, cam, imside=(225,300), use_resize=False, use_batch=False):
        r"""SPair-71k dataset constructor"""
        super(SPairDataset, self).__init__(benchmark, datapath, thres, device, split, imside, use_resize, use_batch)

        self.train_data = open(self.spt_path).read().split('\n') # use large
        self.train_data = self.train_data[:len(self.train_data) - 1]
        self.src_imnames = list(map(lambda x: x.split('-')[1] + '.jpg', self.train_data))
        self.trg_imnames = list(map(lambda x: x.split('-')[2].split(':')[0] + '.jpg', self.train_data))
        self.cls = os.listdir(self.img_path)
        self.cls.sort()
        self.cam = cam

        anntn_files = []
        for data_name in self.train_data:
            anntn_files.append(glob.glob('%s/%s.json' % (self.ann_path, data_name))[0])
        anntn_files = list(map(lambda x: json.load(open(x)), anntn_files))
        self.src_kps = list(map(lambda x: torch.tensor(x['src_kps']), anntn_files))
        self.trg_kps = list(map(lambda x: torch.tensor(x['trg_kps']), anntn_files))
        self.src_bbox = list(map(lambda x: torch.tensor(x['src_bndbox']), anntn_files))
        self.trg_bbox = list(map(lambda x: torch.tensor(x['trg_bndbox']), anntn_files))
        self.cls_ids = list(map(lambda x: self.cls.index(x['category']), anntn_files))

        self.vpvar = list(map(lambda x: torch.tensor(x['viewpoint_variation']), anntn_files))
        self.scvar = list(map(lambda x: torch.tensor(x['scale_variation']), anntn_files))
        self.trncn = list(map(lambda x: torch.tensor(x['truncation']), anntn_files))
        self.occln = list(map(lambda x: torch.tensor(x['occlusion']), anntn_files))

    def __getitem__(self, idx):
        r"""Construct and return a batch for SPair-71k dataset"""
        sample = super(SPairDataset, self).__getitem__(idx)

        sample['src_bbox'] = self.src_bbox[idx]
        sample['trg_bbox'] = self.trg_bbox[idx]
        
        src_mask = self.get_mask(self.src_imnames, idx)
        trg_mask = self.get_mask(self.trg_imnames, idx)
        if src_mask is not None and trg_mask is not None:
            sample['src_mask'] = src_mask
            sample['trg_mask'] = trg_mask

        sample['vpvar'] = self.vpvar[idx]
        sample['scvar'] = self.scvar[idx]
        sample['trncn'] = self.trncn[idx]
        sample['occln'] = self.occln[idx]

        # resize all things
        if self.use_resize:
            sample['src_img'], sample['src_kps'], sample['src_intratio'] = self.resize(sample['src_img'].unsqueeze(0), sample['src_kps'])
            sample['trg_img'], sample['trg_kps'], sample['trg_intratio'] = self.resize(sample['trg_img'].unsqueeze(0), sample['trg_kps'])
            sample['src_bbox'][0::2] *= sample['src_intratio'][1]
            sample['src_bbox'][1::2] *= sample['src_intratio'][0]
            sample['trg_bbox'][0::2] *= sample['trg_intratio'][1]
            sample['trg_bbox'][1::2] *= sample['trg_intratio'][0]
        else:
            sample['src_intratio'] = torch.tensor((1.0, 1.0)) # ratio, (H,W)
            sample['trg_intratio'] = torch.tensor((1.0, 1.0))


        # sample['src_segment'] = self.get_segment(self.src_imnames, idx)
        sample['src_imsize'] = torch.tensor(sample['src_img'].size()) # size, CxHxW
        sample['trg_imsize'] = torch.tensor(sample['trg_img'].size())
        
        sample['pckthres'] = self.get_pckthres(sample) # rescaled pckthres

        if self.use_batch:
            sample['src_kps'] = self.pad_kps(sample['src_kps'], sample['n_pts'])
            sample['trg_kps'] = self.pad_kps(sample['trg_kps'], sample['n_pts'])
                
        return sample

    def get_image(self, img_names, idx):
        r"""Return image tensor"""
        img_name = os.path.join(self.img_path, self.cls[self.cls_ids[idx]], img_names[idx])
        image = self.get_imarr(img_name)
        image = torch.tensor(image.transpose(2, 0, 1).astype(np.float32))

        return image

    def get_mask(self, img_names, idx):
        r"""Return image mask"""
        img_name = os.path.join(self.img_path, self.cls[self.cls_ids[idx]], img_names[idx])
        mask_name = img_name.replace('JPEGImages', self.cam)
        if os.path.exists(mask_name):
            mask = np.array(Image.open(mask_name)) # WxH
        else:
            #print(img_name,mask_name)
            mask = None
        
        return mask

    def get_segment(self, img_names, idx):
        r"""Return segmentation mask"""
        img_name = os.path.join(self.img_path, self.cls[self.cls_ids[idx]], img_names[idx])
        img_name = img_name.replace('JPEGImages', 'Segmentation').replace('jpg', 'png')
        image = self.get_imarr(img_name)
        image = torch.tensor(image.transpose(2, 0, 1).astype(np.float32))

        return image
    
    def pad_kps(self, sample):
        r"""Compute PCK threshold"""
        return super(SPairDataset, self).pad_kps(sample)


    def get_pckthres(self, sample):
        r"""Compute PCK threshold"""
        return super(SPairDataset, self).get_pckthres(sample)

    def get_points(self, pts, idx):
        r"""Return key-points of an image"""
        return super(SPairDataset, self).get_points(pts, idx).t()
