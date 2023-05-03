r"""Implementation of Dynamic Layer Gating (DLG)"""
import torch.nn as nn
import torch
import re
import torch.optim as optim

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

class GradNorm(nn.Module):
    def __init__(self, num_of_task, alpha=1.5):
        super(GradNorm, self).__init__()
        self.num_of_task = num_of_task
        self.alpha = alpha
        self.w = nn.Parameter(torch.ones(num_of_task, dtype=torch.float))
        self.l1_loss = nn.L1Loss()
        self.L_0 = None

    # standard forward pass
    def forward(self, L_t: torch.Tensor):
        # initialize the initial loss `Li_0`
        if self.L_0 is None:
            self.L_0 = L_t.detach() # detach
        # compute the weighted loss w_i(t) * L_i(t)
        self.L_t = L_t
        self.wL_t = L_t * self.w
        # the reduced weighted loss
        self.total_loss = self.wL_t.sum()
        return self.total_loss

    # additional forward & backward pass
    def additional_forward_and_backward(self, grad_norm_weights: nn.Module, optimizer: optim.Optimizer):
        # do `optimizer.zero_grad()` outside
        self.total_loss.backward(retain_graph=True)
        # in standard backward pass, `w` does not require grad
        self.w.grad.data = self.w.grad.data * 0.0

        self.GW_t = []
        for i in range(self.num_of_task):
            # get the gradient of this task loss with respect to the shared parameters
            GiW_t = torch.autograd.grad(
                self.L_t[i], grad_norm_weights.parameters(),
                    retain_graph=True, create_graph=True)
            
            # TODO, check GiW_t size
            # compute the norm
            self.GW_t.append(torch.norm(GiW_t[0] * self.w[i]))
        self.GW_t = torch.stack(self.GW_t) # do not detatch
        self.bar_GW_t = self.GW_t.detach().mean()
        self.tilde_L_t = (self.L_t / self.L_0).detach()
        self.r_t = self.tilde_L_t / self.tilde_L_t.mean()
        grad_loss = self.l1_loss(self.GW_t, self.bar_GW_t * (self.r_t ** self.alpha))
        self.w.grad = torch.autograd.grad(grad_loss, self.w)[0]
        optimizer.step()

        self.GW_ti, self.bar_GW_t, self.tilde_L_t, self.r_t, self.L_t, self.wL_t = None, None, None, None, None, None
        # re-norm
        self.w.data = self.w.data / self.w.data.sum() * self.num_of_task