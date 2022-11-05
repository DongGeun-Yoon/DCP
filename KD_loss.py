import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy.stats import norm
import numpy as np

# NST
class NST(nn.Module):
    '''
    Like What You Like: Knowledge Distill via Neuron Selectivity Transfer
    https://arxiv.org/pdf/1707.01219.pdf
    '''
    def __init__(self):
        super(NST, self).__init__()

    def forward(self, fm_s, fm_t):
        fm_s = list(self.nst(fm) for fm in fm_s)
        fm_t = list(self.nst(fm) for fm in fm_t)
        loss = 0
        for i in range(len(fm_s)):
            loss += (self.poly_kernel(fm_t[i], fm_t[i]).mean() + self.poly_kernel(fm_s[i], fm_s[i]).mean() - 2* self.poly_kernel(fm_s[i], fm_t[i]).mean())
        return loss

    def poly_kernel(self, fm1, fm2):
        fm1 = fm1.unsqueeze(1)
        fm2 = fm2.unsqueeze(2)
        out = (fm1 * fm2).sum(-1).pow(2)
        return out

    def nst(self, fm):
        fm = fm.view(fm.size(0), fm.size(1), -1)
        fm = F.normalize(fm, dim=2)
        return fm

# OFD
class Distiller(nn.Module):
    def __init__(self, t_net, s_net):
        super(Distiller, self).__init__()

        teacher_bns = t_net.get_bn_before_relu()
        margins = [get_margin_from_BN(bn) for bn in teacher_bns]
        for i, margin in enumerate(margins):
            self.register_buffer('margin%d' % (i+1), margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach())

        self.t_net = t_net
        self.s_net = s_net

    def forward(self, x):
        t_feats, t_out = self.t_net.extract_feature(x)
        s_feats, s_out = self.s_net.extract_feature(x)
        b_size = s_out.shape[0]
        feat_num = 4

        loss_distill = 0
        for i in range(feat_num):
            loss_distill += distillation_loss(s_feats[i], t_feats[i].detach(), getattr(self, 'margin%d' % (i+1)))  \
                            / 2 ** (feat_num - i - 1) / b_size

        return loss_distill


def get_margin_from_BN(bn):
    margin = []
    std = bn.weight.data
    mean = bn.bias.data
    for (s, m) in zip(std, mean):
        s = abs(s.item())
        m = m.item()
        if norm.cdf(-m / s) > 0.001:
            margin.append(- s * math.exp(- (m / s) ** 2 / 2) / math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
        else:
            margin.append(-3 * s)

    return torch.FloatTensor(margin).to(std.device)

def distillation_loss(source, target, margin):
    target = torch.max(target, margin)
    loss = torch.nn.functional.mse_loss(source, target, reduction="none")
    loss = loss * ((source > target) | (target > 0)).float()
    return loss.sum()

# SPKD
class SPKD(nn.Module):
    def __init__(self):
        super(SPKD, self).__init__()

    def forward(self, fm_s, fm_t, kd_weight=[1,1]):
        if len(fm_s) == 8:
            spatial = fm_s[:4]
            channel = fm_s[4:]
        else:
            spatial = fm_s
            channel = fm_s

        spatial_t = list(self.spatial_similarity(fm) for fm in fm_t)
        channel_t = list(self.channel_similarity(fm) for fm in fm_t)

        spatial_s = list(self.spatial_similarity(fm) for fm in spatial)
        channel_s = list(self.channel_similarity(fm) for fm in channel)

        loss = 0
        for i in range(len(spatial)):
            loss = kd_weight[0] * torch.mean(torch.pow(spatial_t[i] - spatial_s[i], 2)) \
                + kd_weight[1] * torch.mean(torch.pow(channel_t[i] - channel_s[i],2))
        return loss

    def spatial_similarity(self, fm): # spatial similarity
        fm = fm.view(fm.size(0), fm.size(1),-1)
        norm_fm = fm / (torch.sqrt(torch.sum(torch.pow(fm,2), 1)).unsqueeze(1).expand(fm.shape) + 0.0000001 )
        s = norm_fm.transpose(1,2).bmm(norm_fm)
        s = s.unsqueeze(1)
        return s
    
    def channel_similarity(self, fm): # channel_similarity
        fm = fm.view(fm.size(0), fm.size(1), -1)
        norm_fm = fm / (torch.sqrt(torch.sum(torch.pow(fm,2), 2)).unsqueeze(2).expand(fm.shape) + 0.0000001)
        s = norm_fm.bmm(norm_fm.transpose(1,2))
        s = s.unsqueeze(1)
        return s