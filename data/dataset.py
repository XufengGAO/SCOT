r"""Superclass for semantic correspondence datasets"""
import os

from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F
from model.base.geometry import Geometry

class CorrespondenceDataset(Dataset):
    r"""Parent class of PFPascal, PFWillow, Caltech, and SPair""" # output_image_size = (H, W)
    def __init__(self, benchmark, datapath, thres, split, output_image_size=(256,256), use_resize=False, use_batch=False):
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

        self.use_resize = use_resize
        self.use_batch = use_batch

        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])
                                            ])
        self.split = split

        # To get initialized in subclass constructors
        self.train_data = []
        self.src_imnames = []
        self.trg_imnames = []
        self.cls = []
        self.cls_ids = []
        self.src_kps = []
        self.trg_kps = []

        if benchmark == "caltech":
            self.max_pts = 400
        else:
            self.max_pts = 40
        
        assert isinstance(output_image_size, tuple) or isinstance(output_image_size, int), print("The type of imsize should be tuple or int")
        self.output_image_size = output_image_size # rescale image
        self.benchmark = benchmark

    def __len__(self):
        r"""Returns the number of pairs"""
        return len(self.train_data)

    def __getitem__(self, idx):
        r"""Construct and return a batch"""

        # Image names
        batch = dict()
        batch['src_imname'] = self.src_imnames[idx]
        batch['trg_imname'] = self.trg_imnames[idx]

        # Class of instances in the images
        batch['category_id'] = self.cls_ids[idx]
        batch['pair_class'] = self.cls[batch['category_id']]

        # Image as PIL (original height, original width)
        src_pil = self.get_image(batch['src_imname'])
        trg_pil = self.get_image(batch['trg_imname'])

        batch['src_imsize_ori'] = np.array((src_pil.size[1],src_pil.size[0]))  # h,w
        batch['trg_imsize_ori'] = np.array((trg_pil.size[1],trg_pil.size[0]))

        # Image (original size) as tensor
        batch['src_img'] = self.transform(src_pil) # totensor, CxHxW
        batch['trg_img'] = self.transform(trg_pil)

        # Key-points (original) as tensor
        batch['src_kps'], batch['n_pts'] = self.get_points(self.src_kps, idx)
        batch['trg_kps'], _ = self.get_points(self.trg_kps, idx)

        # The number of pairs in training split
        batch['datalen'] = len(self.train_data)

        return batch

    def get_image(self, img_names):
        r"""Return image tensor"""
        img_name = os.path.join(self.img_path, img_names)
        image = Image.open(img_name).convert('RGB') # WxH

        return image
    
    def get_pckthres(self, sample):
        r"""Compute PCK threshold"""
        if self.thres == 'bbox':
            trg_bbox = sample['trg_bbox']
            return torch.max(trg_bbox[2]-trg_bbox[0], trg_bbox[3]-trg_bbox[1])
        elif self.thres == 'img':
            return torch.tensor(max(sample['trg_img'].size(1), sample['trg_img'].size(2)))
        else:
            raise Exception('Invalid pck evaluation level: %s' % self.thres)

    def get_points(self, pts, idx):
        r"""Returns key-points of an image"""
        return pts[idx], pts[idx].size()[1]
    
    def pad_kps(self, kps, n_pts):
        pad_pts = torch.zeros((2, self.max_pts - n_pts)) - 100 # pad (-1, -1)
        kps = torch.cat([kps, pad_pts], dim=1)
        return kps
    
    def resize(self, img, kps):
        r"""Resize given image with imsize: (1, 3, H, W)"""
        imsize = torch.tensor(img.size()).float()

        if isinstance(self.output_image_size, tuple):
            inter_ratio = (self.output_image_size[0]/imsize[2], self.output_image_size[1]/imsize[3])
            new_size = (self.output_image_size[0], self.output_image_size[1])
        else:
            side_max = torch.max(imsize)
            inter_ratio = (self.output_image_size/side_max, self.output_image_size/side_max) # size reduced to new HxW
            new_size = (int(imsize[2] * inter_ratio[0]), int(imsize[3] * inter_ratio[1]))
            
        img = F.interpolate(img,
                            size=new_size,
                            mode='bilinear',
                            align_corners=False)
        kps[0,:] *= inter_ratio[1]
        kps[1,:] *= inter_ratio[0]
            # , kps, inter_ratio
        return img.squeeze(0), kps, torch.tensor(inter_ratio).float()


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



