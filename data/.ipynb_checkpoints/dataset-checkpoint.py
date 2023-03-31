r"""Superclass for semantic correspondence datasets"""
import os

from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image
import numpy as np

from model.base.geometry import Geometry

class CorrespondenceDataset(Dataset):
    r"""Parent class of PFPascal, PFWillow, Caltech, and SPair""" # imside = (H, W)
    def __init__(self, benchmark, datapath, thres, device, split, imside=(256,256)):
        r"""CorrespondenceDataset constructor"""
        super(CorrespondenceDataset, self).__init__()

        # {Directory name, Layout path, Image path, Annotation path, PCK threshold}
        self.metadata = {
            'pfwillow': ('PF-WILLOW',
                         'test_pairs.csv',
                         '',
                         '',
                         'bbox'),
            'pfpascal': ('PF-PASCAL',
                         '_pairs.csv',
                         'JPEGImages',
                         'Annotations',
                         'img'),
            'caltech':  ('Caltech-101',
                         'test_pairs_caltech_with_category.csv',
                         '101_ObjectCategories',
                         '',
                         ''),
            'spair':   ('SPair-71k',
                        'Layout/large',
                        'JPEGImages',
                        'PairAnnotation',
                        'bbox')
        }

        # Directory path for train, val, or test splits
        base_path = os.path.join(os.path.abspath(datapath), self.metadata[benchmark][0])
        if benchmark == 'pfpascal':
            self.spt_path = os.path.join(base_path, split+'_pairs.csv')
        elif benchmark == 'spair':
            self.spt_path = os.path.join(base_path, self.metadata[benchmark][1], split+'.txt')
        else:
            self.spt_path = os.path.join(base_path, self.metadata[benchmark][1])

        # Directory path for images
        self.img_path = os.path.join(base_path, self.metadata[benchmark][2])

        # Directory path for annotations
        if benchmark == 'spair':
            self.ann_path = os.path.join(base_path, self.metadata[benchmark][3], split)
        else:
            self.ann_path = os.path.join(base_path, self.metadata[benchmark][3])

        # Miscellaneous
        self.thres = self.metadata[benchmark][4] if thres == 'auto' else thres

        if benchmark == "caltech":
            self.max_pts = 400
        else:
            self.max_pts = 40
        self.device = device
        self.imside = torch.tensor(imside) # HxW
        self.range_ts = torch.arange(self.max_pts)
        self.transform  = transforms.Compose([transforms.Resize(size=imside, antialias=True),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])
                                            ])
        self.split = split
        self.benchmark = benchmark

        # To get initialized in subclass constructors
        self.train_data = []
        self.src_imnames = []
        self.trg_imnames = []
        self.cls = []
        self.cls_ids = []
        self.src_kps = []
        self.trg_kps = []

    def __len__(self):
        r"""Returns the number of pairs"""
        return len(self.train_data)

    def __getitem__(self, idx):
        r"""Construct and return a batch"""

        # Image names
        sample = dict()
        sample['src_imname'] = self.src_imnames[idx]
        sample['trg_imname'] = self.trg_imnames[idx]

        # Class of instances in the images
        sample['pair_classid'] = self.cls_ids[idx]
        sample['pair_class'] = self.cls[sample['pair_classid']]

        # Image as numpy
        src_pil = self.get_image(sample['src_imname'])
        trg_pil = self.get_image(sample['trg_imname'])
        sample['src_imsize'] = torch.tensor(src_pil.size).flip(dims=(0,)) # HxW
        sample['trg_imsize'] = torch.tensor(trg_pil.size).flip(dims=(0,))

        # Image as tensor (for trn and val)
        sample['src_img'] = self.transform(src_pil) # totensor, HxW
        sample['trg_img'] = self.transform(trg_pil)

        # Key-points (re-scaled)
        sample['src_kps'], sample['n_pts'] = self.get_points(self.src_kps, idx, sample['src_imsize'])
        sample['trg_kps'], _ = self.get_points(self.trg_kps, idx, sample['trg_imsize'])

        sample['src_ratio'] = (self.imside / sample['src_imsize'])
        sample['trg_ratio'] = (self.imside / sample['trg_imsize'])
        # print(sample['src_imsize'], sample['trg_imsize'] , sample['src_ratio'], sample['trg_ratio'])
 
        # xx
        # The number of pairs in training split
        sample['datalen'] = len(self.train_data)

        return sample

    def get_image(self, img_names):
        r"""Return image tensor"""
        img_name = os.path.join(self.img_path, img_names)
        image = Image.open(img_name).convert('RGB') # WxH
        return image
    
    def get_pckthres(self, batch):
        r"""Computes PCK threshold"""
        if self.thres == 'bbox':
            bbox = batch['trg_bbox'].clone()
            bbox_w = (bbox[2] - bbox[0])
            bbox_h = (bbox[3] - bbox[1])
            pckthres = torch.max(bbox_w, bbox_h)
        elif self.thres == 'img':
            imsize_t = batch['trg_img'].size() # CxHxW
            pckthres = torch.tensor(max(imsize_t[1], imsize_t[2]))
        else:
            raise Exception('Invalid pck threshold type: %s' % self.thres)
        return pckthres.float()

    def get_points(self, pts_list, idx, org_imsize):
        r"""Returns key-points of an image with size of (240,240)"""
        xy, n_pts = pts_list[idx].size()
        # print(pts_list[idx].size())
        pad_pts = torch.zeros((xy, self.max_pts - n_pts)) - 1 # pad (-1, -1)
        # print((self.imside[1] / org_imsize[1]), (self.imside[0] / org_imsize[0]))
        x_crds = pts_list[idx][0] * (self.imside[1] / org_imsize[1]) # w_kps * (300/W)
        y_crds = pts_list[idx][1] * (self.imside[0] / org_imsize[0]) # h_kps * (225/H)
        kps = torch.cat([torch.stack([x_crds, y_crds]), pad_pts], dim=1)

        # return kps.to(self.device), torch.tensor(n_pts).to(self.device)
        return kps, torch.tensor(n_pts)
    
    def match_idx(self, kps, n_pts):
        r"""Samples the nearst feature (receptive field) indices"""

        nearest_idx = find_knn(Geometry.rf_center, kps.t())
        
        nearest_idx -= (self.range_ts >= n_pts).long()
        # print('range_ts', nearest_idx)
        return nearest_idx

def find_knn(db_vectors, qr_vectors):
    r"""Finds K-nearest neighbors (Euclidean distance)"""
    # print("knn", db_vectors.unsqueeze(1).size(), qr_vectors.size())
    # print("knn", db_vectors[-3])
    # (3600, 40, 2), repeated centers for each rep field of each hyperpixel
    db = db_vectors.unsqueeze(1).repeat(1, qr_vectors.size(0), 1)

    # (3600, 40, 2), repeated 40 keypoints
    qr = qr_vectors.unsqueeze(0).repeat(db_vectors.size(0), 1, 1)
    dist = (db - qr).pow(2).sum(2).pow(0.5).t() # (40, 3600)
    # keypoint to each center
    # print("dist", dist.size())
    _, nearest_idx = dist.min(dim=1) #  hyperpixel idx for each keypoint
    # print("nea_idx", nearest_idx.size())
    return nearest_idx

