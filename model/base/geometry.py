"""Provides functions that manipulate boxes and points"""
import torch

from .correlation import Correlation


class Geometry:
    @classmethod  # use function without creating instance
    def initialize(cls, feat_size, device, rfsz, jsz, img_size=256): # feat_size = [60, 60]
        cls.max_pts = 400
        cls.eps = 1e-30
        # rfs, (3600, 4), top-left point (box[:,0], box[:,1]), down-right point (box[:,2], box[:,3])
        cls.rfs, cls.feat_ids = cls.receptive_fields(rfsz, jsz, feat_size, device) # base block, receptive field for each hyperpixel in original image
        cls.rfs = cls.rfs.to(device)
        cls.feat_ids = cls.feat_ids.to(device)
        cls.device = device
        cls.feat_size = feat_size[0]
        # center, (3600, 2)
        cls.rf_center = Geometry.center(cls.rfs) # center for each receptive field in original image
        cls.img_size = img_size

    @classmethod
    def center(cls, box):
        r"""Computes centers, (x, y), of box (N, 4)"""
        x_center = box[:, 0] + torch.div((box[:, 2] - box[:, 0]), 2, rounding_mode='trunc')
        y_center = box[:, 1] + torch.div((box[:, 3] - box[:, 1]), 2, rounding_mode='trunc')
        return torch.stack((x_center, y_center)).t().to(box.device)

    @classmethod
    def receptive_fields(cls, rfsz, jsz, feat_size, device):
        r"""Returns a set of receptive fields (N, 4)"""
        width = feat_size[1]
        height = feat_size[0]
        # print(feat_size)
        feat_ids = torch.tensor(list(range(width))).repeat(1, height).t().repeat(1, 2).to(device)
        feat_ids[:, 0] = torch.tensor(list(range(height))).unsqueeze(1).repeat(1, width).view(-1) # 3600x2

        box = torch.zeros(feat_ids.size()[0], 4).to(device)
        box[:, 0] = feat_ids[:, 1] * jsz - torch.div(rfsz, 2, rounding_mode='trunc')
        box[:, 1] = feat_ids[:, 0] * jsz - torch.div(rfsz, 2, rounding_mode='trunc')
        box[:, 2] = feat_ids[:, 1] * jsz + torch.div(rfsz, 2, rounding_mode='trunc')
        box[:, 3] = feat_ids[:, 0] * jsz + torch.div(rfsz, 2, rounding_mode='trunc')

        return box, feat_ids

    @classmethod
    def gaussian2d(cls, side=7):
        r"""Returns 2-dimensional gaussian filter"""
        dim = [side, side]

        siz = torch.LongTensor(dim)
        sig_sq = (siz.float()/2/2.354).pow(2)
        siz2 = (siz-1)/2

        x_axis = torch.arange(-siz2[0], siz2[0] + 1).unsqueeze(0).expand(dim).float()
        y_axis = torch.arange(-siz2[1], siz2[1] + 1).unsqueeze(1).expand(dim).float()

        gaussian = torch.exp(-(x_axis.pow(2)/2/sig_sq[0] + y_axis.pow(2)/2/sig_sq[1]))
        gaussian = gaussian / gaussian.sum()

        return gaussian

    @classmethod
    def neighbours(cls, box, kps):
        r"""Returns boxes in one-hot format that covers given keypoints"""
        box_duplicate = box.unsqueeze(2).repeat(1, 1, len(kps.t())).transpose(0, 1)
        kps_duplicate = kps.unsqueeze(1).repeat(1, len(box), 1)

        xmin = kps_duplicate[0].ge(box_duplicate[0]) # kps[0] > box[0]
        ymin = kps_duplicate[1].ge(box_duplicate[1]) # kps[1] > box[1] 
        xmax = kps_duplicate[0].le(box_duplicate[2]) # kps[0] < box[2]
        ymax = kps_duplicate[1].le(box_duplicate[3]) # kps[1] < box[3]

        nbr_onehot = torch.mul(torch.mul(xmin, ymin), torch.mul(xmax, ymax)).t()
        n_neighbours = nbr_onehot.sum(dim=1)
        n_points = nbr_onehot.sum(dim=0)
        # nbr_onehot, (8, 3600), each row is one hot (multiple 1s with 0s)
        # each row, sum of 1s means total neighbors/hyperpixels cover this keypoint
        return nbr_onehot, n_neighbours, n_points

    @classmethod
    def transfer_kps(cls, correlation_matrix, kps, n_pts):
        r"""Transfer keypoints by nearest-neighbour assignment"""
        correlation_matrix = Correlation.mutual_nn_filter(correlation_matrix) # refined correleation matrix

        prd_kps = []
        for ct, kpss, np in zip(correlation_matrix, kps, n_pts):

            # 1. Prepare geometries & argmax target indices
            kp = kpss.narrow_copy(1, 0, np) # extract real keypoints
            _, trg_argmax_idx = torch.max(ct, dim=1) # src -> trg hyperpixels ids, highly matched

            geomet = cls.rfs[:, :2].unsqueeze(0).repeat(len(kp.t()), 1, 1) #

            # 2. Retrieve neighbouring source boxes that cover source key-points
            src_nbr_onehot, n_neighbours, _ = cls.neighbours(cls.rfs, kp)
            # print("neighbours", src_nbr_onehot.size(), n_neighbours)

            # 3. Get displacements from source neighbouring box centers to each key-point
            src_displacements = kp.t().unsqueeze(1).repeat(1, len(cls.rfs), 1) - geomet
            src_displacements = src_displacements * src_nbr_onehot.unsqueeze(2).repeat(1, 1, 2).float()

            # 4. Transfer the neighbours based on given correlation matrix
            vector_summator = torch.zeros_like(geomet)
            src_idx = src_nbr_onehot.nonzero()

            trg_idx = trg_argmax_idx.index_select(dim=0, index=src_idx[:, 1])
            vector_summator[src_idx[:, 0], src_idx[:, 1]] = geomet[src_idx[:, 0], trg_idx]
            vector_summator += src_displacements
            prd = (vector_summator.sum(dim=1) / n_neighbours.unsqueeze(1).repeat(1, 2).float()).t()
            
            # print(prd.size(), np, kpss.size())

            # 5. Concatenate pad-points for batch
            pads = (torch.zeros((2, cls.max_pts - np)).to(prd.device) - 1)
            prd = torch.cat([prd, pads], dim=1)
            prd_kps.append(prd)

        return torch.stack(prd_kps)
    
    @classmethod
    def KpsToFlow(cls, src_kpss, trg_kpss, n_ptss):

        flow_maps = []
        for src_kps, trg_kps, n_pts in zip(src_kpss, trg_kpss, n_ptss):
            src_kps = src_kps.t()
            trg_kps = trg_kps.t()

            kp = trg_kps.narrow_copy(0, 0, n_pts) # (11, 2)
            kp_src = src_kps.narrow_copy(0, 0, n_pts)

            src_nbr_onehot, n_neighbours, n_points = cls.neighbours(cls.rfs, kp.t())

            # 11x256, 256
            # print(src_nbr_onehot.size(), n_points.size())
            center = torch.stack(((cls.rfs[:, 0] + cls.rfs[:, 2])/2, (cls.rfs[:, 1] + cls.rfs[:, 3])/2), dim=1)
            center = center.unsqueeze(0).repeat(len(kp), 1, 1) # (11, 256, 2)
            # print(center.size()) # 11x256x2

            src_idx = src_nbr_onehot.nonzero()

            # src_nn = center[src_idx[:,0],src_idx[:,1]] # 44 box for all 11 kps, (kp_id, box_id)
            kp_selected = kp[src_idx[:,0],:] # repeated keypoints 44

            vector_summator = torch.zeros_like(center) # 11x256x2
            # print(vector_summator.size())
            vector_summator[src_idx[:, 0], src_idx[:, 1]] = kp_selected

            n_points_expanded = n_points.unsqueeze(1).repeat(1,2).float() # 256x2
            n_points_expanded[n_points_expanded == 0] = 1

            # multiple keypoints in same box, then sum and average their dist
            # n_points_expanded, number of keypoints in each box
            # reslut, zeros and avg key-coordinates for each box
            # (44x2)
            source_averaged = (vector_summator.sum(dim=0) / n_points_expanded)[src_idx[:,1]]
            # print(source_averaged.size())
            flow = kp_src[src_idx[:,0],:] - source_averaged

            # print(self.feat_ids.size()) # 256x2

            flow_index = cls.feat_ids.index_select(dim=0, index=src_idx[:,1])

            flow_map = torch.zeros(cls.feat_size, cls.feat_size, 2).to(cls.device)
            flow = flow.to(cls.device)
            # print(flow_map.is_cuda, flow_index.is_cuda, flow.is_cuda)
            
            flow_map[flow_index[:,0],flow_index[:,1]] = flow / (cls.img_size // cls.feat_size)

            flow_map = flow_map.permute(2, 0, 1)
        
            flow_maps.append(flow_map.unsqueeze(0))

    
        flow_maps = torch.cat(flow_maps, dim=0)
        return flow_maps       
