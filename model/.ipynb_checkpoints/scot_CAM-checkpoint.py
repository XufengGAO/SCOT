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
from .gating import DynamicFeatureSelection, GradNorm
from .base.geometry import Geometry
import numpy as np

class SCOT_CAM(nn.Module):
    r"""SCOT framework"""
    def __init__(self, args, hyperpixels):
        r"""Constructor for SCOT framework"""
        super(SCOT_CAM, self).__init__()

        # 1. Feature extraction network initialization.
        if args.backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True)
            nbottlenecks = [3, 4, 6, 3]
        elif args.backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True)
            nbottlenecks = [3, 4, 23, 3]
        elif args.backbone == 'fcn101':
            self.backbone = gcv.model_zoo.get_fcn_resnet101_voc(pretrained=True)
            if len(args.cam)==0:
                self.backbone1 = gcv.model_zoo.get_fcn_resnet101_voc(pretrained=True)
                self.backbone1.eval()
            nbottlenecks = [3, 4, 23, 3]
        else:
            raise Exception('Unavailable backbone: %s' % args.backbone)
        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.layer_ids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])

        if len(args.cam) > 0: 
            print('use identity')
            self.backbone.fc = nn.Identity()

        self.backbone
        self.backbone.eval()

        self.hyperpixels = hyperpixels

        # Hyperpixel id and pre-computed jump and receptive field size initialization
        # (the jump and receptive field sizes for 'fcn101' are heuristic values)
        if args.backbone in ['resnet50']:
            self.jsz = torch.tensor([4, 4, 4, 4, 8, 8, 8, 8, 16, 16, 16, 16, 16, 16, 32, 32, 32])
            self.rfsz = torch.tensor([11, 19, 27, 35, 43, 59, 75, 91, 107, 139, 171, 203, 235, 267, 299, 363, 427])
        elif args.backbone in ['resnet101']:
            self.jsz = torch.tensor([4, 4, 4, 4, 8, 8, 8, 8, 16, 16, \
                                     16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, \
                                     16, 16, 16, 16, 32, 32, 32])
            self.rfsz = torch.tensor([11, 19, 27, 35, 43, 59, 75, 91, 107, 139,\
                                      171, 203, 235, 267, 299, 331, 363, 395, 427, 459, 491, 523, 555, 587,\
                                      619, 651, 683, 715, 747, 779, 811, 843, 907, 971])
        elif args.backbone in ['resnet50_ft', 'resnet101_ft']:
            self.jsz = torch.tensor([4, 4, 4, 4, 8, 8, 8, 8, 8, 8])
            self.rfsz = torch.tensor([11, 19, 27, 35, 43, 59, 75, 91, 107, 139])
        else:
            self.jsz = torch.tensor([4, 4, 4, 4, 8, 8, 8, 8, 8, 8])
            self.rfsz = torch.tensor([11, 19, 27, 35, 43, 59, 75, 91, 107, 139])

        # Miscellaneous
        self.hsfilter = geometry.gaussian2d(7).unsqueeze(0).unsqueeze(0).cuda()
        self.benchmark = args.benchmark

        # weighted module
        self.learner = DynamicFeatureSelection(hyperpixels, args.use_xavier, args.weight_thres)
        self.feat_size = (64, 64)

        self.relu = nn.ReLU(inplace=True)

        self.select_all = args.select_all

        self.upsample_size = [int(args.img_side[0]/4), int(args.img_side[1]/4)]


    def forward(self, src_img, trg_img, classmap, src_mask, trg_mask, backbone, model_stage="train"):
        r"""Forward pass"""

        # 1. Update the hyperpixel_ids by checking the weights
        prob = torch.rand(1).item()
        if (prob > self.select_all) and model_stage == "train":
            n_layers = {"resnet50": 17, "resnet101": 34, "fcn101": 34}
            self.hyperpixels = list(range(n_layers[backbone]))
        else:
            self.hyperpixels = self.learner.return_hyperpixel_ids()

        # 2. Update receptive field size
        self.update_rfsz = self.rfsz[self.hyperpixels[0]].to(src_img.device)
        self.update_jsz = self.jsz[self.hyperpixels[0]].to(src_img.device)

        # 3. extract hyperpixel
        src = self.extract_hyperpixel(src_img, classmap, src_mask, backbone)
        trg = self.extract_hyperpixel(trg_img, classmap, trg_mask, backbone)

        return src, trg


    def calculate_sim(self, src_feats, trg_feats, weak_mode=False, bsz=1):
        src_feats = src_feats / (torch.norm(src_feats, p=2, dim=2).unsqueeze(-1)+ 1e-10) # normalized features
        trg_feats = trg_feats / (torch.norm(trg_feats, p=2, dim=2).unsqueeze(-1)+ 1e-10)

        cross_sim = self.relu(torch.bmm(src_feats, trg_feats.transpose(1, 2))) # cross-sim, source->target
        
        if weak_mode:
            src_sim = self.relu(torch.bmm(src_feats[:bsz], src_feats[:bsz].transpose(1, 2)))
            trg_sim = self.relu(torch.bmm(trg_feats[:bsz], trg_feats[:bsz].transpose(1, 2)))   
        else:
            src_sim = None
            trg_sim = None         
    
        del src_feats, trg_feats

        return cross_sim, src_sim, trg_sim

    def calculate_votes(self, src_feats, trg_feats, epsilon, exp2, src_size, trg_size, src_weights=None, trg_weights=None, weak_mode=False, bsz=1):
        cross_sim, src_sim, trg_sim = self.calculate_sim(src_feats, trg_feats, weak_mode, bsz) # only for positive src_sim, trg_sim 
        cross_votes = self.optimal_matching(cross_sim, epsilon, exp2, src_size, trg_size, src_weights, trg_weights)
        if weak_mode:
            src_votes = self.optimal_matching(src_sim, epsilon, exp2, src_size, src_size, src_weights, src_weights)
            trg_votes = self.optimal_matching(trg_sim, epsilon, exp2, trg_size, trg_size, trg_weights, trg_weights)
        else:
            src_votes = None
            trg_votes = None

        return cross_votes, src_votes, trg_votes

    def optimal_matching(self, sim, epsilon, exp2, src_size, trg_size, src_weights=None, trg_weights=None):
            costs = 1 - sim

            if src_weights is not None:
                mus = src_weights / src_weights.sum(dim=1).unsqueeze(-1) # normalize weights
            else:
                mus = (torch.ones((src_size[0],src_size[1]))/src_size[1])

            if trg_weights is not None:
                nus = trg_weights / trg_weights.sum(dim=1).unsqueeze(-1)
            else:
                nus = (torch.ones((src_size[0],trg_size[1]))/trg_size[1])

            del src_weights, trg_weights

            ## ---- <Run Optimal Transport Algorithm> ----
            cnt = 0
            votes = []
            for cost, mu, nu in zip(costs, mus, nus):
                while True: # see Algorithm 1
                    # PI is optimal transport plan or transport matrix.
                    PI = rhm_map.perform_sinkhorn2(cost, epsilon, mu.unsqueeze(-1), nu.unsqueeze(-1)) # 4x4096x4096
                    
                    if not torch.isnan(PI).any():
                        if cnt>0:
                            print(cnt)
                        break
                    else: # Nan encountered caused by overflow issue is sinkhorn
                        epsilon *= 2.0
                        #print(epsilon)
                        cnt += 1

                #exp2 = 1.0 for spair-71k, TSS
                #exp2 = 0.5 # for pf-pascal and pfwillow
                PI = torch.pow(self.relu(src_size[1]*PI), exp2)

                votes.append(PI.unsqueeze(0))

            del mus, nus, sim, costs, PI

            votes = torch.cat(votes, dim=0)

            return votes

    def calculate_votesGeo(self, cross_votes, src_votes, trg_votes, src_imsize, trg_imsize, src_box, trg_box):
        
        cross_votes_geo = self.rhm(cross_votes, src_imsize, trg_imsize, src_box, trg_box)
        src_votes_geo, trg_votes_geo = None, None
        if src_votes is not None:
            src_votes_geo = self.rhm(src_votes, src_imsize, src_imsize, src_box, src_box)
        if trg_votes is not None:
            trg_votes_geo = self.rhm(trg_votes, trg_imsize, trg_imsize, trg_box, trg_box)        

        return cross_votes_geo, src_votes_geo, trg_votes_geo
    
    def rhm(self, votes, src_imsize, trg_imsize, src_box, trg_box):
        with torch.no_grad():
            ncells = 8192
            geometric_scores = []
            nbins_x, nbins_y, hs_cellsize = rhm_map.build_hspace(src_imsize, trg_imsize, ncells)
            bin_ids = rhm_map.hspace_bin_ids(src_imsize, src_box, trg_box, hs_cellsize, nbins_x)
            hspace = src_box.new_zeros((votes.size()[1], nbins_y * nbins_x))

            hbin_ids = bin_ids.add(torch.arange(0, votes.size()[1]).to(votes.device).
                                mul(hspace.size(1)).unsqueeze(1).expand_as(bin_ids))
            for vote in votes:
                new_hspace = hspace.view(-1).index_add(0, hbin_ids.view(-1), vote.view(-1)).view_as(hspace)
                new_hspace = torch.sum(new_hspace, dim=0).to(votes.device)

                # Aggregate the voting results
                new_hspace = F.conv2d(new_hspace.view(1, 1, nbins_y, nbins_x), self.hsfilter, padding=3).view(-1)

                geometric_scores.append((torch.index_select(new_hspace, dim=0, index=bin_ids.view(-1)).view_as(vote)).unsqueeze(0))

            geometric_scores = torch.cat(geometric_scores, dim=0)

        votes = votes * geometric_scores
        del nbins_x, nbins_y, hs_cellsize, bin_ids, hspace, hbin_ids, new_hspace, geometric_scores
        
        return votes

    def extract_cam(self, img, backbone='resnet101'):
        self.hyperpixels = []
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


        hyperfeats, feat_map, fc = self.extract_intermediate_feat(img, return_hp=True, backbone=backbone)
        self.feat_size = (hyperfeats.size()[2], hyperfeats.size()[3])
        # print('feature extrac', hyperfeats.size(), feat_map.size(), fc.size())
        # feature extrac torch.Size([4, 15168, 64, 64]) 
        # torch.Size([4, 2048, 8, 8]) torch.Size([4, 1000])
        
        #TODO: use fixed rfsz = 11, jsz = 4, feat_size for dhpf
        hpgeometry = geometry.receptive_fields(self.update_rfsz, self.update_jsz, hyperfeats.size())
        # print('hpgeometry', hpgeometry.size())
        # torch.Size([4096, 4])

        # flattern each channel: torch.Size([4, 4096, 15168]) = (batch, whole hyperpixels, channels)
        hyperfeats = hyperfeats.view(hyperfeats.size()[0], hyperfeats.size()[1], -1).transpose(1, 2)
        # print(hyperfeats.size(), len(hyperfeats))

        # len return 1st dim
        weights = torch.ones(len(hyperfeats), hyperfeats.size()[1]).to(hyperfeats.device)
        # print(weights.size())

        # weight points, CAM weighter pixels importance
        if classmap in [1]: 
            if mask is None:
                # get CAM mask
                if backbone=='fcn101':
                    mask = self.get_FCN_map(img, feat_map, fc, sz=(img.size(2),img.size(3)))
                else:
                    mask = self.get_CAM_multi(img, feat_map, fc, sz=(img.size(2),img.size(3)), top_k=2)                 
            scale = 1.0
            del feat_map, fc
            
            hpos = geometry.center(hpgeometry) # 4096 2
            # print("hpos", hpos.size(), hpos[:], Geometry.rf_center.size(), Geometry.rf_center[:])

            # mask: Bx256x256 -> hselect: Bx4096
            hselect = mask[:, hpos[:,1].long(),hpos[:,0].long()].to(hpos.device)
            # print(torch.max(hselect), torch.min(hselect), hselect.size())

            weights = 0.5*torch.ones(hselect.size()).to(hpos.device)
            weights[hselect>0.4*scale] = 0.8  # weighted CAM with gamma and beta
            weights[hselect>0.5*scale] = 0.9
            weights[hselect>0.6*scale] = 1.0

            del hpos, hselect
            
        # print('no-clasmap', weights.size())
        # print(img.size())
        # print(img.size()[2:][::-1], weights.unsqueeze(-1).size())

        
        results = {'box':hpgeometry, 'feats':hyperfeats, 'imsize':img.size()[2:][::-1], 'weights':weights}
        del hpgeometry, hyperfeats, weights

        return results

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
        channels = []
        with torch.no_grad():
            # Layer 0
            feat = self.backbone.conv1.forward(img)
            del img
            feat = self.backbone.bn1.forward(feat)
            feat = self.backbone.relu.forward(feat)
            feat = self.backbone.maxpool.forward(feat)
            if 0 in self.hyperpixels:
                feats.append(feat) # scaled feats
                channels.append(feat.size()[1])

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

                if hid + 1 in self.hyperpixels:
                    feats.append(feat)
                    channels.append(feat.size()[1])

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
        

        # scaled the features
        for i,  (hyper_id, hyper_feat) in enumerate(zip(self.hyperpixels, feats)):
            feats[i] = self.learner(hyper_id, hyper_feat)
        feats = torch.cat(feats, dim=1)  # 4-dim, BCHW

        # print(feats.size(), feats.requires_grad)
        # torch.Size([2, 15168, 64, 64]) True

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

            del output_cam, cam

        map_list = torch.stack(map_list,dim=0) # 3xBx240x240
        sum_cam = map_list.sum(dim=0) # Bx240x240
        sum_cam_max = sum_cam.view(bz,-1).max(dim=-1,keepdim=True)[0].unsqueeze(-1)
        norm_cam = sum_cam / (sum_cam_max+1e-10) # Bx240x240
        # print(map_list.size(), sum_cam.size(), sum_cam_max.size(), norm_cam.size())
        # transform = T.ToPILImage()
        # for idx, outputcam in enumerate(norm_cam):
        #     imgm = transform(outputcam)
        #     file_name = "{}".format(idx)
        #     imgm.save("/home/xufeng/Documents/EPFL_Course/sp_code/SCOT/img/{}.png".format(file_name))
        del map_list, sum_cam, sum_cam_max

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