r"""Normalization functions"""
import torch.nn.functional as F
import torch



def feat_normalize(x, interp_size):
    r"""L2-normalizes given 2D feature map after interpolation"""
    x = F.interpolate(x, interp_size, mode='bilinear', align_corners=True)
    return x.pow(2).sum(1).view(x.size(0), -1)


def l1normalize(x):
    r"""L1-normalization"""
    vector_sum = torch.sum(x, dim=2, keepdim=True)
    vector_sum[vector_sum == 0] = 1.0
    return x / vector_sum


def linearnormalize(x):
    r"""linear normalization"""
    vectore_max = torch.max(x, dim=1, keepdim=True)
    vectore_min = torch.max(x, dim=1, keepdim=True)

    return (x - vectore_min)/(vectore_max - vectore_min + 1e-30)


def unit_gaussian_normalize(x):
    r"""Make each (row) distribution into unit gaussian"""
    correlation_matrix = x - x.mean(dim=2).unsqueeze(2).expand_as(x)

    with torch.no_grad():
        standard_deviation = correlation_matrix.std(dim=2)
        standard_deviation[standard_deviation == 0] = 1.0
    correlation_matrix /= standard_deviation.unsqueeze(2).expand_as(correlation_matrix)

    return correlation_matrix
