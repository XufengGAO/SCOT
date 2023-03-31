"""Implementation of optimal transport+geometric post-processing (Hough voting)"""

import math

import torch.nn.functional as F
import torch
import torch.nn as nn
from . import geometry


relu = nn.ReLU()

def perform_sinkhorn(C,epsilon,mu,nu,a=[],warm=False,niter=10,tol=10e-9):
    """Main Sinkhorn Algorithm"""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if not warm:
        a = torch.ones((C.shape[0], C.shape[1], 1)) / torch.ones((C.shape[0], C.shape[1], 1)).sum(dim=1).unsqueeze(1)
        a = a.to(device)
        # TODO  a = a.cuda()

    K = torch.exp(-C/epsilon)

    Err = torch.zeros((niter,2)).to(device)
    # Err = torch.zeros((niter,2)).cuda()
    for i in range(niter):
        b = nu/torch.bmm(K.transpose(1,2), a)
        # K=4x4096x4096, a=4x4096x1, b=4x4096x1, mu=4x4096x1
        # print("k", K.size(), a.size(), b.size(), mu.size(), torch.norm(a*(torch.bmm(K, b)) - mu, dim=1, p=1).mean())

        if i%2==0:
            Err[i,0] = torch.norm(a*(torch.bmm(K, b)) - mu, dim=1, p=1).mean()
            if i>0 and (Err[i,0]) < tol:
                break

        a = mu / torch.bmm(K, b)

        if i%2==0:
            Err[i,1] = torch.norm(b*(torch.bmm(K.transpose(1,2), a)) - nu, dim=1, p=1).mean()
            if i>0 and (Err[i,1]) < tol:
                break
        # print("a", torch.diag_embed(a.squeeze(-1)).size(), a[0,:2,:],torch.diag_embed(a.squeeze(-1))[0,:2,:2] )
        PI = torch.bmm(torch.bmm(torch.diag_embed(a.squeeze(-1)),K), torch.diag_embed(b.squeeze(-1)))
    
    # del a; del b; del K
    return PI,mu,nu,Err


def appearance_similarity(src_feats, trg_feats, exp1=3):
    r"""Semantic appearance similarity (exponentiated cosine)"""
    src_feat_norms = torch.norm(src_feats, p=2, dim=1).unsqueeze(1)
    trg_feat_norms = torch.norm(trg_feats, p=2, dim=1).unsqueeze(0)
    sim = torch.matmul(src_feats, trg_feats.t()) / \
          torch.matmul(src_feat_norms, trg_feat_norms)
    sim = torch.pow(relu(sim), exp1)

    return sim


def appearance_similarityOT(src_feats, trg_feats, exp1=1.0, exp2=1.0, eps=0.05, src_weights=None, trg_weights=None):
    r"""Semantic Appearance Similarity"""
    # st_weights = src_weights.mm(trg_weights.t())
    # compute correleation maps
    # print("src feat", src_feats.requires_grad, trg_feats.requires_grad

    # print("feat", src_feats.size(), trg_feats.size()) # torch.Size([4, 4096, 15168])
    src_feat_norms = torch.norm(src_feats, p=2, dim=2).unsqueeze(2) # [4, 4096, 1]
    trg_feat_norms = torch.norm(trg_feats, p=2, dim=2).unsqueeze(1) # [4, 1, 4096]
    # print("Norm", src_feat_norms.size(), trg_feat_norms.size()) # torch.Size([3750, 1])

    correleation = torch.bmm(src_feats, trg_feats.transpose(1, 2))
    # print(correleation.size()) # [4, 4096, 4096]

    sim = correleation / (torch.bmm(src_feat_norms, trg_feat_norms) + 0.001)
    sim = torch.pow(relu(sim), 1.0) # clamp
    # print("PI", sim.size(), sim.max())
    # print("similarity map", sim.size()) # [4, 4096, 4096]

    #sim = sim*st_weights
    cost = 1-sim
    
    n1 = src_feats.size()[1]
    # TODO, may change later
    # mu = (torch.ones((n1,))/n1).cuda()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # mu = (torch.ones((n1,))/n1).to(device)

    mu = src_weights / src_weights.sum(dim=1).unsqueeze(1) # normalize weights
    # TODO  nu = (torch.ones((n2,))/n2).cuda()
    # nu = (torch.ones((n2,))/n2).to(device)
    nu = trg_weights / trg_weights.sum(dim=1).unsqueeze(1)
    ## ---- <Run Optimal Transport Algorithm> ----
    #mu = mu.unsqueeze(1)
    #nu = nu.unsqueeze(1)

    # print(mu.size(), nu.size())


    # with torch.no_grad():
    epsilon = eps
    cnt = 0
    while True: # see Algorithm 1
        # PI is optimal transport plan or transport matrix.
        PI,_,_,_ = perform_sinkhorn(cost, epsilon, mu, nu) # 4x4096x4096

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
    PI = torch.pow(relu(PI), exp2)

    return PI, sim



def hspace_bin_ids(src_imsize, src_box, trg_box, hs_cellsize, nbins_x):
    r"""Compute Hough space bin id for the subsequent voting procedure"""
    src_ptref = torch.tensor(src_imsize, dtype=torch.float).to(src_box.device)
    src_trans = geometry.center(src_box)
    trg_trans = geometry.center(trg_box)
    xy_vote = (src_ptref.unsqueeze(0).expand_as(src_trans) - src_trans).unsqueeze(2).\
                  repeat(1, 1, len(trg_box)) + \
              trg_trans.t().unsqueeze(0).repeat(len(src_box), 1, 1)

    bin_ids = (xy_vote / hs_cellsize).long()

    return bin_ids[:, 0, :] + bin_ids[:, 1, :] * nbins_x


def build_hspace(src_imsize, trg_imsize, ncells):
    r"""Build Hough space where voting is done"""
    hs_width = src_imsize[0] + trg_imsize[0]
    hs_height = src_imsize[1] + trg_imsize[1]
    hs_cellsize = math.sqrt((hs_width * hs_height) / ncells)
    nbins_x = int(hs_width / hs_cellsize) + 1
    nbins_y = int(hs_height / hs_cellsize) + 1

    return nbins_x, nbins_y, hs_cellsize


def rhm(src_hyperpixels, trg_hyperpixels, hsfilter, sim, exp1, exp2, eps, ncells=8192):
    r"""Regularized Hough matching"""
    # Unpack hyperpixels
    src_hpgeomt, src_hpfeats, src_imsize, src_weights = src_hyperpixels
    trg_hpgeomt, trg_hpfeats, trg_imsize, trg_weights = trg_hyperpixels

    # Prepare for the voting procedure
    if sim in ['cos', 'cosGeo']:
        votes = appearance_similarity(src_hpfeats, trg_hpfeats, exp1)
    if sim in ['OT', 'OTGeo']:
        votes, sim = appearance_similarityOT(src_hpfeats, trg_hpfeats, exp1, exp2, eps, src_weights, trg_weights)
    if sim in ['OT', 'cos', 'cos2']:
        return votes
    # print("votes:", votes.size(), votes.requires_grad, torch.max(votes), torch.min(votes))
    # print("imsize", src_imsize, trg_imsize, src_hpgeomt.size(), ncells)

    geometric_scores = []
    # with torch.no_grad():
    nbins_x, nbins_y, hs_cellsize = build_hspace(src_imsize, trg_imsize, ncells)
    bin_ids = hspace_bin_ids(src_imsize, src_hpgeomt, trg_hpgeomt, hs_cellsize, nbins_x)
    hspace = src_hpgeomt.new_zeros((len(src_hpgeomt), nbins_y * nbins_x))

    # Proceed voting
    hbin_ids = bin_ids.add(torch.arange(0, len(src_hpgeomt)).to(src_hpgeomt.device).
                           mul(hspace.size(1)).unsqueeze(1).expand_as(bin_ids))
    for vote in votes:
        new_hspace = hspace.view(-1).index_add(0, hbin_ids.view(-1), vote.detach().clone().view(-1)).view_as(hspace)
        new_hspace = torch.sum(new_hspace, dim=0)

        # Aggregate the voting results
        new_hspace = F.conv2d(new_hspace.view(1, 1, nbins_y, nbins_x),
                        hsfilter.unsqueeze(0).unsqueeze(0), padding=3).view(-1)

        geometric_scores.append(torch.index_select(new_hspace, dim=0, index=bin_ids.view(-1)).view_as(vote.detach().clone()))

    geometric_scores = torch.stack(geometric_scores, dim=0)
    # print("geometric scores", geometric_scores.size()) # 4x4096x4096

    votes_geo = votes * geometric_scores
    return sim, votes, votes_geo


