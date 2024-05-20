import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import SimpleITK as sitk

class Get_gradient_loss(nn.Module):
    def __init__(self):
        super(Get_gradient_loss, self).__init__()
        
        kernel_w,kernel_h,kernel_d = torch.zeros((3,3,3)),torch.zeros((3,3,3)),torch.zeros((3,3,3))
        
        kernel_w[0,1,1]=-1
        kernel_w[2,1,1]=1
        
        kernel_h[1,1,0]=-1
        kernel_h[1,1,2]=1
        
        kernel_d[1,0,1]=-1
        kernel_d[1,2,1]=1
       

        self.weight_w = nn.Parameter(data = kernel_w, requires_grad = False).view(1,1,3,3,3).cuda()
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).view(1,1,3,3,3).cuda()
        self.weight_d = nn.Parameter(data = kernel_d, requires_grad = False).view(1,1,3,3,3).cuda()
        
        self.mse_loss=nn.MSELoss()

    def get_gradient(self, x):  # x: (b,1,w,h,d)  k: (1,1,3,3,3)  output: (b,1,w,h,d)
    
        #print(x.shape)
        g_w=F.conv3d(x,self.weight_w,padding=1)
        g_h=F.conv3d(x,self.weight_h,padding=1)
        g_d=F.conv3d(x,self.weight_d,padding=1)
        
        output = torch.sqrt(torch.pow(g_w, 2) + torch.pow(g_h, 2)+ torch.pow(g_d, 2)+ 1e-6)
        #p3d=(1,1,1,1,1,1)
        #output=F.pad(output,p3d,mode='replicate')
        return output
        
    def get_mask(self,gradient,thre=0.8):
        #threshold=torch.quantile(gradient,torch.tensor(thre).cuda())
        threshold=np.quantile(gradient.cpu().numpy(),thre)
        mask=(gradient>threshold).float()
        return mask
        
    def forward(self,x1,x2):
        #print(x1.shape,x2.shape)
        gradient_1=self.get_gradient(x1)
        gradient_2=self.get_gradient(x2)
        
        gradient_loss=self.mse_loss(gradient_1,gradient_2)
        return gradient_loss
