# modified from: https://github.com/thstkdgus35/EDSR-PyTorch
from argparse import Namespace
import torch.nn as nn
from .NLSA import NonLocalSparseAttention

def conv_3d(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

        
class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size,bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class EDSR_3d(nn.Module):
    def __init__(self, args):
        super(EDSR_3d, self).__init__()
        
        # define head module
        m_head = [conv_3d(args.in_channel, args.n_feats, args.kernel_size)]
        # define body module
        m_body = [ResBlock(conv_3d, args.n_feats, args.kernel_size, res_scale=args.res_scale)
                  for _ in range(args.n_resblocks)]
        
        self.add_tail=args.add_tail
        if self.add_tail:
            m_tail = [conv_3d(args.n_feats, args.in_channel, args.kernel_size)]
            self.tail = nn.Sequential(*m_tail)
        else:
            m_body.append(conv_3d(args.n_feats, args.n_feats, args.kernel_size))
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

    def forward(self, x):
        if self.add_tail:
            res = x+self.body(self.head(x))
        else:
            x = self.head(x)
            res = self.body(x)+x
        if self.add_tail:
            res=self.tail(res)
        return res


def make_edsr_baseline(n_resblocks=12, n_feats=64, res_scale=1,add_tail=False):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale
    
    args.kernel_size = 3
    args.in_channel = 1
    args.add_tail=add_tail

    return EDSR_3d(args)


