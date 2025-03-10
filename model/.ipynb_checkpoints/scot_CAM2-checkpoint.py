"""Implementation of : Semantic Correspondence as an Optimal Transport Problem"""

from functools import reduce
from operator import add
import torch.nn.functional as F
import torch
import gluoncv as gcv
from . import geometry
from . import rhm_map
from . import resnet
import torch.nn as nn
from . import gating
from .base.geometry import Geometry
import numpy as np

class SCOT_CAM(nn.Module):
    r"""SCOT framework"""
    def __init__(self, backbone, hyperpixel_ids, benchmark, device, cam="", use_xavier=False, weight_thres=0.05, select_all=0.85):
        r"""Constructor for SCOT framework"""
        super(SCOT_CAM, self).__init__()

        # 1. Feature extraction network initialization.
        if backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True)
            nbottlenecks = [3, 4, 6, 3]
        elif backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True)
            nbottlenecks = [3, 4, 23, 3]
        elif backbone == 'fcn101':
            self.backbone = gcv.models.get_fcn_resnet101_voc(pretrained=True).to(device).pretrained
            if len(cam)==0:
                self.backbone1 = gcv.models.get_fcn_resnet101_voc(pretrained=True)
                self.backbone1.eval()
            nbottlenecks = [3, 4, 23, 3]
        elif backbone == 'resnet34':
            self.backbone = resnet.resnet34(pretrained=True)
            nbottlenecks = [3, 4, 6, 3]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)
        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.layer_ids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        if len(cam) > 0: 
            print('use identity')
            self.backbone.fc = nn.Identity()
        self.backbone.to(device)
        self.backbone.eval()

        self.hyperpixel_ids = hyperpixel_ids

        # Hyperpixel id and pre-computed jump and receptive field size initialization
        # (the jump and receptive field sizes for 'fcn101' are heuristic values)
        if backbone in ['resnet50']:
            self.jsz = torch.tensor([4, 4, 4, 4, 8, 8, 8, 8, 16, 16, 16, 16, 16, 16, 32, 32, 32]).to(device)
            self.rfsz = torch.tensor([11, 19, 27, 35, 43, 59, 75, 91, 107, 139, 171, 203, 235, 267, 299, 363, 427]).to(device)
        elif backbone in ['resnet101']:
            self.jsz = torch.tensor([4, 4, 4, 4, 8, 8, 8, 8, 16, 16, \
                                     16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, \
                                     16, 16, 16, 16, 32, 32, 32]).to(device)
            self.rfsz = torch.tensor([11, 19, 27, 35, 43, 59, 75, 91, 107, 139,\
                                    171, 203, 235, 267, 299, 331, 363, 395, 427, 459, 491, 523, 555, 587,\
                                      619, 651, 683, 715, 747, 779, 811, 843, 907, 971]).to(device)
        elif backbone in ['resnet50_ft', 'resnet101_ft']:
            self.jsz = torch.tensor([4, 4, 4, 4, 8, 8, 8, 8, 8, 8]).to(device)
            self.rfsz = torch.tensor([11, 19, 27, 35, 43, 59, 75, 91, 107, 139]).to(device)
        else:
            self.jsz = torch.tensor([4, 4, 4, 4, 8, 8, 8, 8, 8, 8]).to(device)
            self.rfsz = torch.tensor([11, 19, 27, 35, 43, 59, 75, 91, 107, 139]).to(device)

        # Miscellaneous
        self.hsfilter = geometry.gaussian2d(7).to(device)
        self.device = device
        self.benchmark = benchmark

        # weighted module
        self.learner = gating.DynamicFeatureSelection(hyperpixel_ids, use_xavier, weight_thres).to(device)
        self.feat_size = (64, 64)

        self.select_all = select_all

    def forward(self, src_img, trg_img, sim, exp1, exp2, eps, classmap, src_mask, trg_mask, backbone, training="trn", trg_cen=False):
        r"""Forward pass"""

        # 1. Update the hyperpixel_ids by checking the weights
        prob = torch.rand(1).item()
        if (prob > self.select_all) and training == "train":
            n_layers = {"resnet50": 17, "resnet101": 34, "fcn101": 34}
            self.hyperpixel_ids = list(range(n_layers[backbone]))
        else:
            self.hyperpixel_ids = self.learner.return_hyperpixel_ids()

        # 2. Update receptive field size
        self.update_rfsz = self.rfsz[self.hyperpixel_ids[0]]
        self.update_jsz = self.jsz[self.hyperpixel_ids[0]]

        # 3. extract hyperpixel
        src_hpgeomt, src_hpfeats, src_imsize, src_weights = self.extract_hyperpixel(src_img, classmap, src_mask, backbone)
        trg_hpgeomt, trg_hpfeats, trg_imsize, trg_weights = self.extract_hyperpixel(trg_img, classmap, trg_mask, backbone)

        # 4. scaled corelation maps
        correlation, src_norm, trg_norm = 0, 0, 0
        src_hps = []
        trg_hps = []
        for hyper_id, src_hp, trg_hp in zip(self.hyperpixel_ids, src_hpfeats, trg_hpfeats):
            src_hp = src_hp.view(src_hp.size(0), src_hp.size(1), -1).transpose(1,2)
            trg_hp = trg_hp.view(trg_hp.size(0), trg_hp.size(1), -1)

            # 1. F^tw -> 1D
            # src_hps.append(self.learner(hyper_id, src_hp))
            # trg_hps.append(self.learner(hyper_id, src_hp))
            
            correlation += self.learner(hyper_id, torch.bmm(src_hp, trg_hp))

            src_norm +=  src_hp.pow(2).sum(2, keepdim=True)  # [4, 4096, 1]
            trg_norm += trg_hp.pow(2).sum(1, keepdim=True) # [4, 1, 4096]

        # src_hps = torch.cat(src_hps, dim=2).transpose(1,2)
        # trg_hps = torch.cat(trg_hps, dim=1)
        
        src_norm = src_norm.pow(0.5)
        trg_norm = trg_norm.pow(0.5)

        sim = nn.ReLU()(correlation / (torch.bmm(src_norm, trg_norm) + 1e-10))

        # 5. optimal matching
        costs = 1-sim

        bz = sim.size()[0]
        n1 = sim.size()[1]
        mus = (torch.ones((bz,n1))/n1).to(self.device)
        if src_weights is not None:
            mus = src_weights / src_weights.sum(dim=1).unsqueeze(-1) # normalize weights

        n2 = sim.size()[2]
        nus = (torch.ones((bz,n2))/n1).to(self.device)
        if trg_weights is not None:
            nus = trg_weights / trg_weights.sum(dim=1).unsqueeze(-1)

        epsilon = eps
        cnt = 0
        votes = []
        for cost, mu, nu in zip(costs, mus, nus):

            # print('cost', cost.size()) # [4096, 4096]
            # print('mu/nu', mu.size(), nu.size()) # [4096]
            

            while True: # see Algorithm 1
                # PI is optimal transport plan or transport matrix.
                PI,_,_,_ = rhm_map.perform_sinkhorn2(cost, epsilon, mu.unsqueeze(-1), nu.unsqueeze(-1)) # 4x4096x4096
                
                #PI = sinkhorn_stabilized(mu, nu, cost, reg=epsilon, numItermax=50, method='sinkhorn_stabilized', cuda=True)
                if not torch.isnan(PI).any():
                    if cnt>0:
                        print(cnt)
                    break
                else: # Nan encountered caused by overflow issue is sinkhorn
                    epsilon *= 2.0
                    #print(epsilon)
                    cnt += 1

            PI = n1*PI # re-scale PI 
            #exp2 = 1.0 for spair-71k, TSS
            #exp2 = 0.5 # for pf-pascal and pfwillow
            PI = torch.pow(nn.ReLU()(PI), exp2)

            votes.append(PI.unsqueeze(0))

        votes = torch.cat(votes, dim=0)

        # 6. rhm
        with torch.no_grad():
            ncells = 8192
            geometric_scores = []
            nbins_x, nbins_y, hs_cellsize = rhm_map.build_hspace(src_imsize, trg_imsize, ncells)
            bin_ids = rhm_map.hspace_bin_ids(src_imsize, src_hpgeomt, trg_hpgeomt, hs_cellsize, nbins_x)
            hspace = src_hpgeomt.new_zeros((votes.size()[1], nbins_y * nbins_x))

            hbin_ids = bin_ids.add(torch.arange(0, votes.size()[1]).to(src_hpgeomt.device).
                                mul(hspace.size(1)).unsqueeze(1).expand_as(bin_ids))
            for vote in votes:
                new_hspace = hspace.view(-1).index_add(0, hbin_ids.view(-1), vote.view(-1)).view_as(hspace)
                new_hspace = torch.sum(new_hspace, dim=0)

                # Aggregate the voting results
                new_hspace = F.conv2d(new_hspace.view(1, 1, nbins_y, nbins_x),
                                self.hsfilter.unsqueeze(0).unsqueeze(0), padding=3).view(-1)

                geometric_scores.append((torch.index_select(new_hspace, dim=0, index=bin_ids.view(-1)).view_as(vote)).unsqueeze(0))

            geometric_scores = torch.cat(geometric_scores, dim=0)
            
        votes_geo = votes * geometric_scores

        if training == "test":
            return votes_geo, src_hpgeomt, trg_hpgeomt
        else:
            src_center = geometry.center(src_hpgeomt)
            if trg_cen:
                trg_center = geometry.center(trg_hpgeomt)
            else:
                trg_center = None
            
            return sim, votes, votes_geo, src_hpgeomt, trg_hpgeomt, self.feat_size, src_center, trg_center, src_hps, trg_hps

    def extract_cam(self, img, backbone='resnet101'):
        self.hyperpixel_ids = []
        feat_map, fc = self.extract_intermediate_feat(img, return_hp=False, backbone=backbone)
        mask = self.get_CAM_multi2(img, feat_map, fc, sz=(img.size(2),img.size(3)), top_k=2)
        # print(mask.size(), torch.max(mask), torch.min(mask))

        return mask

    def extract_hyperpixel(self, img, classmap, mask, backbone="resnet101"):
        r"""Given image, extract desired list of hyperpixels \
            Input:
                classmap == args.classmap, 0 for beamsearch
            Output:
                hpgeometry: receptive field (N,4)
                hyperfeats: (batch, total_pixels, channel)
                img.size()[1:][::-1],: torch.Size([W, H])
                weights: weight importance for each pixel according to CAM, (num_pixels, 1)
        """

        # reutrn a hyper-list
        hyperfeats, feat_map, fc = self.extract_intermediate_feat(img, return_hp=True, backbone=backbone)
        self.feat_size = (hyperfeats[0].size()[2], hyperfeats[0].size()[3])
        # print('feature extrac', hyperfeats.size(), feat_map.size(), fc.size())
        # feature extrac torch.Size([4, 15168, 64, 64]) 
        # torch.Size([4, 2048, 8, 8]) torch.Size([4, 1000])
        
        hpgeometry = geometry.receptive_fields(self.update_rfsz, self.update_jsz, hyperfeats[0].size()).to(self.device)
        # print('hpgeometry', hpgeometry.size())
        # torch.Size([4096, 4])

        # flattern each channel: torch.Size([4, 4096, 15168]) = (batch, whole hyperpixels, channels)
        # hyperfeats = hyperfeats.view(hyperfeats[0].size()[0], hyperfeats.size()[1], -1).transpose(1, 2)
        # print(hyperfeats.size(), len(hyperfeats))


        # len return 1st dim
        # weight: torch.Size([2, 3750])
        weights = torch.ones(len(hyperfeats[0]), hyperfeats[0].size()[2]*hyperfeats[0].size()[3]).to(hyperfeats[0].device)
        
        # weight points, CAM weighter pixels importance
        if classmap in [1]: 
            if mask is None:
                # get CAM mask
                if backbone=='fcn101':
                    mask = self.get_FCN_map(img, feat_map, fc, sz=(img.size(2),img.size(3)))
                else:
                    mask = self.get_CAM_multi(img, feat_map, fc, sz=(img.size(2),img.size(3)), top_k=2)
                    # mask = []
                    # for im, fm, fc in zip(img, feat_map, fc):
                    #     ms = self.get_CAM_multi2(im.unsqueeze(0), fm.unsqueeze(0), fc.unsqueeze(0), sz=(img.size(2),img.size(3)), top_k=2)
                    #     mask.append(ms)
                    # if len(mask) > 1:
                    #     mask = torch.stack(mask, dim=0)
                    # else:
                    #     mask = mask[0].unsqueeze(0)

                    # print(torch.sum((masks- mask)>1e-3))                    
            scale = 1.0
            
            hpos = geometry.center(hpgeometry) # 4096 2
            # print("hpos", hpos.size(), hpos[:], Geometry.rf_center.size(), Geometry.rf_center[:])

            # mask: Bx256x256 -> hselect: Bx4096
            hselect = mask[:, hpos[:,1].long(),hpos[:,0].long()].to(hpos.device)
            # print(torch.max(hselect), torch.min(hselect), hselect.size())

            weights = 0.5*torch.ones(hselect.size()).to(hpos.device)
            weights[hselect>0.4*scale] = 0.8  # weighted CAM with gamma and beta
            weights[hselect>0.5*scale] = 0.9
            weights[hselect>0.6*scale] = 1.0

            del hpos
            del hselect
            
        # print('no-clasmap', weights.size())
        # print(img.size())
        # print(img.size()[2:][::-1], weights.unsqueeze(-1).size())

        return hpgeometry, hyperfeats, img.size()[2:][::-1], weights


    def extract_intermediate_feat(self, img, return_hp=True, backbone='resnet101'):
        r"""Extract desired a list of intermediate features \
            Input:
                return_hp: return hyperpixels, and rfsz, jsz \
            Output:
                feats[0]: feature stack, size = feature[0], 3-dim, changes later if use batch
                rfsz: receptive field size for id[0]
                jsz: jump size for id[0]
                feat_map: feature map before gloabl avg-pool
                fc: final output, after backbone.fc
        """

        feats = []

        with torch.no_grad():
            # Layer 0
            feat = self.backbone.conv1.forward(img)
            feat = self.backbone.bn1.forward(feat)
            feat = self.backbone.relu.forward(feat)
            feat = self.backbone.maxpool.forward(feat)
            if 0 in self.hyperpixel_ids:
                feats.append(feat.clone()) # scaled feats

            # Layer 1-4
            for hid, (bid, lid) in enumerate(zip(self.bottleneck_ids, self.layer_ids)):
                res = feat
                feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
                feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
                feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
                feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
                feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
                feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
                feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
                feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)
                
                if bid == 0:
                    res = self.backbone.__getattr__('layer%d' % lid)[bid].downsample.forward(res)
                
                feat += res

                if hid + 1 in self.hyperpixel_ids:
                    feats.append(feat.clone())

                feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

            # Global Average Pooling feature map
            feat_map = feat  # feature map before gloabl avg-pool
            if backbone!='fcn101':
                x = self.backbone.avgpool(feat)
                x = torch.flatten(x, 1)
                fc = self.backbone.fc(x)  # fc output
            else:
                fc = None

            if not return_hp: # only return final outputs, feat_map and fc
                return feat_map,fc
            
            # torch.Size([4, 3, 240, 240]) torch.Size([4, 2048, 8, 8]) torch.Size([4, 1000])
            # print(img.size(), feat_map.size(), fc.size())

            # Up-sample & concatenate features to construct a hyperimage
            for idx, hyper_feat in enumerate(feats):
                if idx == 0:
                    continue
                # upsampling deep features
                feats[idx] = F.interpolate(hyper_feat, tuple(feats[0].size()[2:]), None, 'bilinear', True)
        
        # torch.Size([2, C, 50, 75])

        # return 3-dim tensor, cause B=1
        return feats, feat_map, fc
    

    def get_CAM(self, feat_map, fc, sz, top_k=2):
        logits = F.softmax(fc, dim=1)
        scores, pred_labels = torch.topk(logits, k=top_k, dim=1)

        pred_labels = pred_labels[0]
        bz, nc, h, w = feat_map.size()

        output_cam = []
        for label in pred_labels:
            cam = self.backbone.fc.weight[label,:].unsqueeze(0).mm(feat_map.view(nc,h*w))
            cam = cam.view(1,1,h,w)
            cam = F.interpolate(cam, (sz[0],sz[1]), None, 'bilinear', True)[0,0] # HxW
            cam = (cam-cam.min()) / cam.max()
            output_cam.append(cam)
        output_cam = torch.stack(output_cam,dim=0) # kxHxW
        output_cam = output_cam.max(dim=0)[0] # HxW

        return output_cam


    def get_CAM_multi2(self, img, feat_map, fc, sz, top_k=2):
        scales = [1.0,1.5,2.0]
        map_list = []
        for scale in scales:
            if scale>1.0:
                if scale*scale*sz[0]*sz[1] > 800*800:
                    scale = min(800/sz[0],800/sz[1])
                    scale = min(1.5,scale)
                img = F.interpolate(img, (int(scale*sz[0]),int(scale*sz[1])), None, 'bilinear', True) # 1x3xHxW
                feat_map, fc = self.extract_intermediate_feat(img,return_hp=False)

            logits = F.softmax(fc, dim=1)
            scores, pred_labels = torch.topk(logits, k=top_k, dim=1)
            pred_labels = pred_labels[0]
            bz, nc, h, w = feat_map.size()

            output_cam = []
            for label in pred_labels:
                cam = self.backbone.fc.weight[label,:].unsqueeze(0).mm(feat_map.view(nc,h*w))
                cam = cam.view(1,1,h,w)
                cam = F.interpolate(cam, (sz[0],sz[1]), None, 'bilinear', True)[0,0] # HxW
                cam = (cam-cam.min()) / cam.max()
                output_cam.append(cam)
            output_cam = torch.stack(output_cam,dim=0) # kxHxW
            output_cam = output_cam.max(dim=0)[0] # HxW
            
            map_list.append(output_cam)
        map_list = torch.stack(map_list,dim=0)
        sum_cam = map_list.sum(0)
        norm_cam = sum_cam / (sum_cam.max()+1e-5)

        return norm_cam
    
    
    def get_CAM_multi(self, img, feat_map, fc, sz, top_k=2):
        # img = Bx3x256x256
        # featmap = Bx2048x8x8
        # fc = Bx1000
        # sz = 256x256
        scales = [1.0,1.5,2.0]
        map_list = []
        for scale in scales:
            if scale>1.0:
                if scale*scale*sz[0]*sz[1] > 800*800:
                    scale = min(800/sz[0],800/sz[1])
                    scale = min(1.5,scale)
                img = F.interpolate(img, (int(scale*sz[0]),int(scale*sz[1])), None, 'bilinear', True) # Bx3xHxW
                feat_map, fc = self.extract_intermediate_feat(img,return_hp=False)

            logits = F.softmax(fc, dim=1)
            _, pred_labels = torch.topk(logits, k=top_k, dim=1) # Bx2
            bz, nc, h, w = feat_map.size()
            output_cam = []

            # print(self.backbone.fc.weight.size()) # 1000x2048
            cam = self.backbone.fc.weight[pred_labels,:].bmm(feat_map.view(bz, nc, h*w)) # Bx2048x64
            cam = cam.view(bz,-1,h,w) # Bx2x8x8
            cam = F.interpolate(cam, (sz[0],sz[1]), None, 'bilinear', True) # Bx2x240x240
            
            cam_min, _ = torch.min(cam.view(bz, top_k, -1), dim=-1, keepdim=True) #Bx2x1
            cam_max, _ = torch.max(cam.view(bz, top_k, -1), dim=-1, keepdim=True)
            cam_min = cam_min.unsqueeze(-1)  #Bx2x1x1
            cam_max = cam_max.unsqueeze(-1)
            cam = (cam-cam_min)/cam_max # Bx2x240x240
            output_cam = cam.max(dim=1)[0] # Bx240x240

            map_list.append(output_cam)

        map_list = torch.stack(map_list,dim=0) # 3xBx240x240
        sum_cam = map_list.sum(dim=0) # Bx240x240
        sum_cam_max = sum_cam.view(bz,-1).max(dim=-1,keepdim=True)[0].unsqueeze(-1)
        norm_cam = sum_cam / (sum_cam_max+1e-5) # Bx240x240
        # print(map_list.size(), sum_cam.size(), sum_cam_max.size(), norm_cam.size())
        # transform = T.ToPILImage()
        # for idx, outputcam in enumerate(norm_cam):
        #     imgm = transform(outputcam)
        #     file_name = "{}".format(idx)
        #     imgm.save("/home/xufeng/Documents/EPFL_Course/sp_code/SCOT/img/{}.png".format(file_name))
        
        return norm_cam


    def get_FCN_map(self, img, feat_map, fc, sz):
        #scales = [1.0,1.5,2.0]
        scales = [1.0]
        map_list = []
        for scale in scales:
            if scale*scale*sz[0]*sz[1] > 1200*800:
                scale = 1.5
            img = F.interpolate(img, (int(scale*sz[0]),int(scale*sz[1])), None, 'bilinear', True) # 1x3xHxW
            #feat_map, fc = self.extract_intermediate_feat(img,return_hp=False,backbone='fcn101')
            feat_map = self.backbone1.evaluate(img)
            
            predict = torch.max(feat_map, 1)[1]
            mask = predict-torch.min(predict)
            mask_map = mask / torch.max(mask)
            mask_map = F.interpolate(mask_map.unsqueeze(0).double(), (sz[0],sz[1]), None, 'bilinear', True)[0,0] # HxW
    
        return mask_map
    
    def parameters(self):
        return self.learner.parameters()

    def state_dict(self):
        return self.learner.state_dict()
    
    def load_backbone(self, state_dict):
        self.backbone.load_state_dict(state_dict, strict=False)

    def load_state_dict(self, state_dict):
        self.learner.load_state_dict(state_dict)

    def eval(self):
        self.learner.eval()

    def train(self):
        self.learner.train()