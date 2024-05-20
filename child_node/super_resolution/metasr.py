import torch
import math
import torch.nn as nn

def default_conv(in_channels, out_channels, kernel_size, bias=True):
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


class MetaSR(nn.Module):
    def __init__(self, n_feats=64, n_colors=1,n_resblocks=8,res_scale=1,\
                 conv=default_conv,kernel_size = 3,act = nn.ReLU(True)):
        super(MetaSR, self).__init__()

        self.P2W = Pos2Weight(inC=n_feats)

        # define head module
        m_head = [conv(n_colors, n_feats, kernel_size)]
        # define body module
        m_body = [ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
                  for _ in range(n_resblocks)]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        
    def get_feat(self,x):
        x = self.head(x)
        feat = self.body(x)
        feat += x   # (batch_size, inC, inH, inW, inD)
        return feat
        
    def inference(self, feat, projection_coord, offset_vector): # coord (batch_size,sample_num,3) vector (batch_size,sample_num,6)
        
        #padding
        pad = nn.ReplicationPad3d(padding=1)
        matrix = pad(feat)

        # following steps are used to get projected_feat, shape=(batch_size,sample_num,inC*kernel_size*kernel_size)
        vx_list=[-1,0,1]
        vy_list=[-1,0,1]
        vz_list=[-1,0,1]
        batch_size,sample_num=projection_coord.shape[:2]

        feature_list=[]  #
        for vx in vx_list:
            for vy in vy_list:
                for vz in vz_list:
                    offset=torch.tensor([vx,vy,vz]).view(1,1,3).expand(batch_size,sample_num,3).cuda(device=feat.device)
                    coord=projection_coord+offset
                    feature=self.projection(matrix,coord)  # (batch_size,sample_num,64)
                    feature_list.append(feature)
        feat_mat=torch.stack(feature_list,dim=-1).contiguous().view(batch_size*sample_num,1,-1)       # (batch_size,sample_num,64,k^3)

        # (batch_size,sample_num,inC *outC* kernel_size*kernel_size*kernel_size)
        local_weight = self.P2W(offset_vector)
        # (batch_size,sample_num,inC *kernel_size*kernel_size*kernel_size,1)
        local_weight = local_weight.contiguous().view(batch_size*sample_num, -1, 1)

        pred=torch.bmm(feat_mat,local_weight).view(batch_size,sample_num,1)

        return {'pred':pred}

    def projection(self,matrix,coord):
        # matrix (b,64,H,W,D)  coord(b,N,3)
        b, N = coord.shape[:2]
        ind = torch.arange(0, b).view(b, 1).expand(b, N).contiguous().view(-1)
        # return (b,N,64)
        return matrix[ind, :, coord[:, :, 0].view(-1), coord[:, :, 1].view(-1), coord[:, :, 2].view(-1)].view(b,N,-1)
        
    def forward(self, x, projection_coord, offset_vector):
        feat=self.get_feat(x)
        out=self.inference(feat, projection_coord, offset_vector)
        return out
    
class Pos2Weight(nn.Module):
    def __init__(self,inC=64, kernel_size=3, outC=1):
        super(Pos2Weight,self).__init__()
        self.inC = inC
        self.kernel_size=kernel_size
        self.outC = outC
        self.meta_block=nn.Sequential(
            nn.Linear(6,256),
            nn.ReLU(inplace=True),
            nn.Linear(256,self.kernel_size**3*self.inC*self.outC)
        )
    def forward(self,x):

        output = self.meta_block(x)
        return output

