r"""PF-PASCAL dataset"""
import os

import scipy.io as sio
import pandas as pd
import numpy as np
import torch

from .dataset import CorrespondenceDataset
from PIL import Image


class PFPascalDataset(CorrespondenceDataset):
    r"""Inherits CorrespondenceDataset"""
    def __init__(self, benchmark, datapath, thres, device, split, cam, imside=(200,300), use_resize=False, use_batch=False):
        r"""PF-PASCAL dataset constructor"""
        super(PFPascalDataset, self).__init__(benchmark, datapath, thres, device, split, imside, use_resize, use_batch)

        self.train_data = pd.read_csv(self.spt_path) # dataframe
        self.src_imnames = np.array(self.train_data.iloc[:, 0])
        self.trg_imnames = np.array(self.train_data.iloc[:, 1])
        self.cls = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                    'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person',
                    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        self.cls_ids = self.train_data.iloc[:, 2].values.astype('int') - 1
        self.cam = cam  # point 1

        if split == 'trn': # trn.csv file inclues column 'flip'
            self.flip = self.train_data.iloc[:, 3].values.astype('int')
        self.src_kps = [] # list of tensor keypoints
        self.trg_kps = [] 
        self.src_bbox = [] # list of tensor bbx (x11, x12, x21, x22)
        self.trg_bbox = []

        # loop over each pair
        for src_imname, trg_imname, cls in zip(self.src_imnames, self.trg_imnames, self.cls_ids):
            # read annotation files
            src_anns = os.path.join(self.ann_path, self.cls[cls],
                                    os.path.basename(src_imname))[:-4] + '.mat'
            trg_anns = os.path.join(self.ann_path, self.cls[cls],
                                    os.path.basename(trg_imname))[:-4] + '.mat'

            src_kp = torch.tensor(read_mat(src_anns, 'kps')).float()
            trg_kp = torch.tensor(read_mat(trg_anns, 'kps')).float()
            src_box = torch.tensor(read_mat(src_anns, 'bbox')[0].astype(float))
            trg_box = torch.tensor(read_mat(trg_anns, 'bbox')[0].astype(float))

            src_kps = []
            trg_kps = []
            for src_kk, trg_kk in zip(src_kp, trg_kp):
                # if kp is nan, just ignore
                if len(torch.isnan(src_kk).nonzero()) != 0 or \
                        len(torch.isnan(trg_kk).nonzero()) != 0:
                    continue
                else:
                    src_kps.append(src_kk)
                    trg_kps.append(trg_kk)
            self.src_kps.append(torch.stack(src_kps).t()) # stacked kp for one image, size = (2, num_kp)
            self.trg_kps.append(torch.stack(trg_kps).t())
            self.src_bbox.append(src_box) # bbx consists of 4 numbers
            self.trg_bbox.append(trg_box)

        self.src_imnames = list(map(lambda x: os.path.basename(x), self.src_imnames))
        self.trg_imnames = list(map(lambda x: os.path.basename(x), self.trg_imnames))

    def __getitem__(self, idx):
        r"""Construct and return a batch for PF-PASCAL dataset"""
        sample = super(PFPascalDataset, self).__getitem__(idx)

        # Object bounding-box (list of tensors)
        sample['src_bbox'] = self.src_bbox[idx]
        sample['trg_bbox'] = self.trg_bbox[idx]

        # print(sample['src_bbox'], sample['trg_bbox'])
        
        src_mask = self.get_mask(self.src_imnames, idx)# only possible when cam exists
        trg_mask = self.get_mask(self.trg_imnames, idx)
        if src_mask is not None and trg_mask is not None:
            sample['src_mask'] = src_mask
            sample['trg_mask'] = trg_mask
            
        # Horizontal flip of key-points when training (no training in HyperpixelFlow)
        if self.split == 'trn' and self.flip[idx]: # width - current x-axis
            # sample['src_kps'][0] = sample['src_img'].size()[2] - sample['src_kps'][0]
            # sample['trg_kps'][0] = sample['trg_img'].size()[2] - sample['trg_kps'][0]
            self.horizontal_flip(sample)
            sample['flip'] = 1
        else:
            sample['flip'] = 0



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


        sample['src_imsize'] = torch.tensor(sample['src_img'].size()) # rescaled size, CxHxW
        sample['trg_imsize'] = torch.tensor(sample['trg_img'].size())

        sample['pckthres'] = self.get_pckthres(sample) # rescaled pckthres


        # for key, value in sample.items():
        #     if key in ['src_img', 'trg_img']:
        #         print(key, value.size())
        #     else:
        #         print(key, value)
        # x

        if self.use_batch:
            sample['src_kps'] = self.pad_kps(sample['src_kps'], sample['n_pts'])
            sample['trg_kps'] = self.pad_kps(sample['trg_kps'], sample['n_pts'])

        return sample

    def pad_kps(self, sample):
        r"""Compute PCK threshold"""
        return super(PFPascalDataset, self).pad_kps(sample)

    
    def horizontal_flip(self, sample):
        tmp = sample['src_bbox'][0].clone()
        sample['src_bbox'][0] = sample['src_img'].size(2) - sample['src_bbox'][2]
        sample['src_bbox'][2] = sample['src_img'].size(2) - tmp

        tmp = sample['trg_bbox'][0].clone()
        sample['trg_bbox'][0] = sample['trg_img'].size(2) - sample['trg_bbox'][2]
        sample['trg_bbox'][2] = sample['trg_img'].size(2) - tmp

        sample['src_kps'][0] = sample['src_img'].size(2) - sample['src_kps'][0]
        sample['trg_kps'][0] = sample['trg_img'].size(2) - sample['trg_kps'][0]

        sample['src_img'] = torch.flip(sample['src_img'], dims=(2,))
        sample['trg_img'] = torch.flip(sample['trg_img'], dims=(2,))
   

    def get_mask(self, img_names, idx):
        r"""Return image mask"""
        img_name = os.path.join(self.img_path, img_names[idx])
        mask_name = img_name.replace('/JPEGImages', '-'+self.cam) # TODO: prepared cam folder
        if os.path.exists(mask_name):
            mask = np.array(Image.open(mask_name)) # WxH
        else:
            #print(img_name,mask_name)
            mask = None
        
        return mask


def read_mat(path, obj_name):
    r"""Read specified objects from Matlab data file, (.mat)"""
    mat_contents = sio.loadmat(path)
    mat_obj = mat_contents[obj_name]

    return mat_obj
