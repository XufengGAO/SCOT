r"""Implementation of Dynamic Layer Gating (DLG)"""
import torch.nn as nn
import torch

class DynamicFeatureSelection(nn.Module):
    def __init__(self, feat_ids = [], use_xavier=False, thres=0.05):
        super(DynamicFeatureSelection, self).__init__()
        self.feat_ids = feat_ids
        
        # define a learnable weight w in (0,1)
        self.layerweight = nn.Parameter(torch.zeros(1, len(self.feat_ids)))

        self.thres = thres

        if use_xavier:
            nn.init.xavier_normal_(self.layerweight.data)

    def forward(self, idx, feat):
        layerweight_norm = self.layerweight[0, idx].sigmoid()
        return feat * layerweight_norm 
    
    def return_hyperpixel_ids(self):
        with torch.no_grad():
            layerweight_norm = self.layerweight[0].sigmoid().data

        return (layerweight_norm > self.thres).nonzero().view(-1).tolist()