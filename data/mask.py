r"""PF-PASCAL dataset"""
import os

import scipy.io as sio
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

class MaskDataset(Dataset):
    r"""Inherits CorrespondenceDataset"""
    def __init__(self, benchmark, datapath, device, split, imside=(256,256), use_resize=False):
        r"""PF-PASCAL dataset constructor"""
        super(MaskDataset, self).__init__()

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
        self.use_resize = use_resize
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])
                                            ])
        self.device = device
        self.split = split

        self.train_data = []
        self.src_imnames = []

        assert isinstance(imside, tuple) or isinstance(imside, int), print("The type of imsize should be tuple or int")
        self.imside = imside # rescale image
        self.benchmark = benchmark

        self.train_data = pd.read_csv(self.spt_path) # dataframe
        self.img_imnames = np.array(self.train_data.iloc[:, 0])

        self.src_imnames = list(map(lambda x: os.path.basename(x), self.src_imnames))

    def __len__(self):
        r"""Returns the number of pairs"""
        return len(self.train_data)
    
    def __getitem__(self, idx):
        r"""Construct and return a batch for PF-PASCAL dataset"""

        # Image names
        sample = dict()
        sample['img_imname'] = self.img_imnames[idx]

        # Image as PIL
        src_pil = self.get_image(sample['img_imname'])

        # Image (original size) as tensor
        sample['src_img'] = self.transform(src_pil) # totensor, CxHxW

        if self.use_resize:
            imsize = torch.tensor(sample['src_img'].unsqueeze(0).size()).float()

            if isinstance(self.imside, tuple):
                inter_ratio = (self.imside[0]/imsize[2], self.imside[1]/imsize[3])
                new_size = (self.imside[0], self.imside[1])
            else:
                side_max = torch.max(imsize)
                inter_ratio = (self.imside/side_max, self.imside/side_max) # size reduced to new HxW
                new_size = (int(imsize[2] * inter_ratio[0]), int(imsize[3] * inter_ratio[1]))
                
            sample['src_img'] = F.interpolate(sample['src_img'].unsqueeze(0),
                                size=new_size,
                                mode='bilinear',
                                align_corners=False)
            
            sample['src_img'] = sample['src_img'].squeeze(0)

        # The number of pairs in training split
        sample['datalen'] = len(self.train_data)
                
        return sample
    

    def get_image(self, img_names):
        r"""Return image tensor"""
        img_name = os.path.join(self.img_path, img_names)
        image = Image.open(img_name).convert('RGB') # WxH

        return image
