import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .edsr import conv_3d, make_edsr_baseline
from .mlp import MLP
from .NLSA import NonLocalSparseAttention
from .utils import make_coord,input_matrix_wpn,to_pixel_samples
from .AttentionLayer import AttentionLayer

class NLSALayer(nn.Module):
    def __init__(self, n_feats):
        super(NLSALayer, self).__init__()
        
        self.atten=NonLocalSparseAttention(channels=n_feats)  
        self.relu = nn.ReLU()
        self.conv = nn.Conv3d(n_feats,n_feats,  kernel_size=3,padding=1, bias=True)
             
    def forward(self, x):
        x=self.atten(x)
        a = self.conv(self.relu(x))
        return x+a
        
class FFNLayer(nn.Module):
    def __init__(self, n_feats):
        super(FFNLayer, self).__init__()
        
        self.fc1 = nn.Linear(n_feats,n_feats)
        self.fc2 = nn.Linear(n_feats,n_feats)
        
        self.norm = nn.LayerNorm(n_feats)
        
    def forward(self, x):

        a = self.fc1(F.elu(self.fc2(x)))
        x = self.norm(x + a)
        
        return x
        
class SAINR(nn.Module):
    def __init__(self, n_resblocks=8, n_feats=64, win_size=(7, 7, 2),layerType='FBLA', dilation=1,\
                     add_res=False,add_NLSA=False,add_branch=False):
        super().__init__()
        self.add_res=add_res
        self.add_NLSA=add_NLSA
        self.add_branch=add_branch

        self.encoder = make_edsr_baseline(n_resblocks, n_feats)
        
        if self.add_NLSA:    
            self.NLSAlayer=NLSALayer(n_feats)       
        if self.add_branch:
            self.mask_predictor=nn.Linear(n_feats,2)

        self.attentionLayer=AttentionLayer(n_feats,win_size,layerType,dilation)
        self.imnet = MLP(in_dim=n_feats, out_dim=1, hidden_list=[256, 256, 256, 256])
        
    def get_feat(self,inp):     
        feat =self.encoder(inp)  # feat (b,c,w/2,h/2,d/2)
        if self.add_NLSA:  
            feat=self.NLSAlayer(feat)
        return feat
        
    def inference(self, inp, feat, hr_coord,proj_coord):  # inp (b,1,w/2,h/2,d/2)  # hr_coord (b,w*h*d,3) # proj_coord (b,w*h*d,3)
                
        n_feats=feat.shape[1]

        bs, sample_q = hr_coord.shape[:2]
        q_feat = F.grid_sample(feat, hr_coord.flip(-1).view(bs, 1, 1, sample_q, 3), mode='bilinear',
                               align_corners=True)[:, :, 0, 0, :].permute(0, 2, 1)  # q_feat (b,w*h*d,c)=(b,n,c)
        
        if not self.add_branch:
            mask=torch.cat([torch.zeros((bs, sample_q,1)),torch.ones((bs, sample_q,1))],dim=-1).to(inp.device)

            ##### if you do not want to use the inter-slice attention, enable the following code while disabling the above code
            # mask=torch.cat([torch.ones((bs, sample_q,1)),torch.zeros((bs, sample_q,1))],dim=-1).to(inp.device)
        else:  
            mask_p=self.mask_predictor(q_feat)  # mask (b,n,2)
            mask=F.gumbel_softmax(mask_p,tau=1,hard=True,dim=2)  # return 2-dim one-hot tensor
   
        # branch 1
        idx_easy= torch.nonzero(mask[:,:,0].view(-1)).squeeze(1)   # idx_easy (m1)
        if torch.sum(mask[:,:,0])>0:
            feat_easy=torch.index_select(q_feat.contiguous().view(-1,n_feats),0,idx_easy)   # feat_easy (m1,c)
            #feat_easy=self.FFNLayer1(feat_easy)
            pred_easy=self.imnet(feat_easy) # pred_easy (m1,1)
         
        # branch 2  
        idx_difficult= torch.nonzero(mask[:,:,1].view(-1)).squeeze(1)   # idx_difficult (m2)
        if torch.sum(mask[:,:,1])>0:
            pred_difficult=[]
            for i in range(bs):            
                idx_each= torch.nonzero(mask[i,:,1].view(-1)).squeeze(1)  # idx_each (m2/)      
                hr_coord_each=torch.index_select(hr_coord[i],0,idx_each).unsqueeze(0)  # hr_coord_each(1,m2/,3)
                proj_coord_each=torch.index_select(proj_coord[i],0,idx_each).unsqueeze(0)  # proj_coord_each (1,m2/,3)
                
                q_feat_each=torch.index_select(q_feat[i],0,idx_each).unsqueeze(0)  # feat_each (1,m2/,c)
                feat_each=self.attentionLayer(q_feat_each,feat[i].unsqueeze(0), proj_coord_each, hr_coord_each).squeeze(0)
                #feat_each=self.FFNLayer2(feat_each)
                pred_each=self.imnet(feat_each)  # (m2/,1)
                pred_difficult.append(pred_each)
            
            pred_difficult=torch.cat(pred_difficult,dim=0)  # pred_difficult (m2,1)
        # combine and scatter
        pred=torch.zeros(bs*sample_q, 1).cuda()
        if torch.sum(mask[:,:,0])==0:
            idx=idx_difficult.unsqueeze(1)
            pred_shuffled=pred_difficult
        elif torch.sum(mask[:,:,1])==0:
            idx=idx_easy.unsqueeze(1)
            pred_shuffled=pred_easy
        else:     
            idx=torch.cat([idx_easy,idx_difficult]).unsqueeze(1)   # idx (n,1)
            pred_shuffled=torch.cat([pred_easy,pred_difficult])
        pred=pred.type_as(pred_shuffled)
        pred.scatter_(0, idx,pred_shuffled)
        pred=pred.unsqueeze(0).view(bs,sample_q,1)
        #print(pred.shape)
        if self.add_res:
            ip = F.grid_sample(inp, hr_coord.flip(-1).view(bs, 1, 1, sample_q, 3), mode='bilinear', align_corners=True)[
                 :, :, 0, 0, :].permute(0, 2, 1)  # ip (b,w*h*d,1)
            
            pred += ip
            
        return {"pred":pred,
                "mask":mask[:,:,1]}
       
    def forward(self,inp, hr_coord,proj_coord):
        feat=self.get_feat(inp)
        return self.inference(inp,feat, hr_coord,proj_coord)
           

class ArSSR(nn.Module):
    def __init__(self,n_resblocks=8,n_feats=64,local_ensemble=False, feat_unfold=False, cell_decode=False):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        self.imnet,self.encoder=self.initialize(n_resblocks,n_feats)
        
        
    def initialize(self,n_resblocks,n_feats):
        imnet=MLP(in_dim=n_feats+3,out_dim=1,hidden_list=[256, 256, 256, 256])
        encoder=make_edsr_baseline(n_resblocks, n_feats) 
        return imnet,encoder
        
    def get_feat(self, inp):
        feat = self.encoder(inp)
        return feat
        
    def inference(self, feat, hr_coord):  
         
        bs,sample_q=hr_coord.shape[:2]
        q_feat=F.grid_sample(feat,hr_coord.flip(-1).view(bs,1,1,sample_q,3),mode='bilinear', align_corners=False)[:, :, 0, 0, :] .permute(0, 2, 1)  # q_feat (b,w*h*d,c)        
        x = torch.cat([q_feat, hr_coord], dim=-1)  #x (b,w*h*d,c+3) c=64             
        pred=self.imnet(x) # pred (b,w*h*d,1)
        return {'pred':pred}
        
    def forward(self, inp, hr_coord):  # inp (b,1,w/2,h/2,d/2)  # hr_coord (b,w*h*d,3) 
        feat=self.get_feat(inp)   # feat (b,c,w/2,h/2,d/2)   
        return self.inference(feat,hr_coord)


    
if __name__ == '__main__':
    import scipy.ndimage
    from thop import profile
    import SimpleITK as sitk

    scale=2
    crop_hr=torch.randn(256,256,scale+1)
    hr_coord, hr_value,proj_coord = to_pixel_samples(crop_hr,scale=(1,1,scale))
    
    #sample_q=64*64*10
    #sample_lst = np.random.choice(hr_coord.shape[0], sample_q, replace=False)
    #hr_coord = hr_coord[sample_lst].unsqueeze(0).cuda()  # self.sample_q,3
    #proj_coord=proj_coord[sample_lst].unsqueeze(0).cuda()   #  self.sample_q,3
    
    hr_coord = hr_coord.unsqueeze(0).cuda()
    proj_coord=proj_coord.unsqueeze(0).cuda()
    model = LIIF(add_res=True,add_branch=True).cuda()
    st=torch.load('0629/theta_0.8/_train_edsr-baseline-liif/epoch-last.pth',map_location='cpu')
    model.load_state_dict(st['model'])
    # calculate flops for each branch
    img=sitk.GetArrayFromImage(sitk.ReadImage('/hpc/data/home/bme/v-wangxin/skull_stripped/100206_3T_T1w_MPR1.nii.gz'))
    idxs=range(0,img.shape[-1],scale)
    img=img[:,:,idxs]
    
    flops_list=[]
    for i in range(img.shape[-1]-1):
        slice1=torch.FloatTensor(scipy.ndimage.zoom(img[:,:,i], \
            (256/320,256/320), order=3 , mode='nearest'))
        slice2=torch.FloatTensor(scipy.ndimage.zoom(img[:,:,i+1], \
            (256/320,256/320), order=3 , mode='nearest')) 
        inp=torch.stack([slice1,slice2],dim=-1).unsqueeze(0).unsqueeze(0).cuda()      
        flops, params = profile(model, inputs=(inp,hr_coord,proj_coord))
        print('FLOPs = ' + str(flops/1000**3) + 'G')
        # exit(0)
        flops_list.append(flops)
    flops=sum(flops_list)/len(flops_list)
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    # print('Params = ' + str(params/1000**2) + 'M')
    
    
    






