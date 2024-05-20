from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .utils import input_matrix_wpn, make_coord, normalize

class TestSet(Dataset):
    def __init__(self,patch_list):       
        self.patch_list = patch_list

    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, idx):
        crop_lr=self.patch_list[idx]
        return {
            'inp': torch.FloatTensor(crop_lr).unsqueeze(0)
        }
        
def test(ckpt_name,model,img,ratio,axis=0):
   
    # Image=sitk.ReadImage(path)   # LR image
    # img=sitk.GetArrayFromImage(Image)
    
    # target_spacing=(spacing[0],spacing[0],slice_spacing)
    # spacing_ratio=(1,1,spacing[2]/target_spacing[2])
    spacing_ratio=(1,1,ratio)
    
    if axis==0:
        img = img.transpose(1,2,0)
    elif axis==1:
        img=img.transpose(0,2,1)
 
    inp = normalize(img).astype(np.float64)  # w*h*d 
    shape=(inp.shape[0],inp.shape[1],(inp.shape[2]-1)*spacing_ratio[2]+1)
    sr=[]
    if ckpt_name.startswith('EDSR'):
        inp=torch.nn.functional.interpolate(torch.FloatTensor(inp[None, None]), \
            size=shape, mode="trilinear", align_corners=True)[0, 0].numpy()
        print("inp.shape",inp.shape)
        pw,ph,pd=inp.shape[0],inp.shape[1],16*ratio+1
        sw,sh,sd=32,32,16
    
        patch_list=[]
        for i in range(0,shape[0]-pw+1,sw):     
            for j in range(0,shape[1]-ph+1,sh):  
                for k in range(0,shape[2]-pd+1,sd):
                    patch=inp[i:i+pw,j:j+ph,k:k+pd]
                    patch_list.append(patch)
    
        test_set=TestSet(patch_list)
        test_loader=DataLoader(test_set, batch_size=1, shuffle=False, \
            num_workers=8, pin_memory=True)
     
        pred_list=[]
        for batch in tqdm(test_loader, leave=False, desc='test'):
            for k, v in batch.items():
                batch[k] = v.cuda()
            pred = model(batch['inp'])
            pred_list.append(pred.squeeze(1).cpu().numpy().astype(np.float64))
        pred_list=np.concatenate(pred_list,axis=0)
        idx=0
        sr=np.zeros(shape=shape)
        num=np.zeros(shape=shape)
        max_i,max_j,max_k=0,0,0
        for i in range(0,shape[0]-pw+1,sw): 
            for j in range(0,shape[1]-ph+1,sh): 
                for k in range(0,shape[2]-pd+1,sd):
                    sr[i:i+pw,j:j+ph,k:k+pd]+=pred_list[idx]
                    num[i:i+pw,j:j+ph,k:k+pd]+=1
                    max_i,max_j,max_k=i+pw,j+ph,k+pd
                    idx+=1
        sr[:max_i,:max_j,:max_k]=sr[:max_i,:max_j,:max_k]/num[:max_i,:max_j,:max_k]
        sr[max_i:,max_j:,max_k:]=inp[max_i:,max_j:,max_k:]
        # sr=model(inp.cuda()).squeeze().cpu().numpy().astype(np.float64)
    elif ckpt_name.startswith('ArSSR'):
        inp=torch.FloatTensor(inp).view(1,1,*inp.shape).cuda()
        feat=model.get_feat(inp)
        hr_coord=make_coord(shape)  # (w,h,d,3)
        for i in range(shape[2]):
            hr_coord_d=hr_coord[:,:,i,:].view(-1,3).unsqueeze(0).cuda()
            output=model.inference(feat, hr_coord_d)
            pred=output['pred'].view(shape[0],shape[1]).cpu().numpy().astype(np.float64)
            sr.append(pred)
        sr=np.stack(sr,axis=-1)
    elif ckpt_name.startswith('SAINR'):
        inp=torch.FloatTensor(inp).view(1,1,*inp.shape).cuda()
        feat=model.get_feat(inp)
        hr_coord=make_coord(shape)  # (w,h,d,3)
        proj_coord=input_matrix_wpn(*shape, scale=spacing_ratio, flatten=False)['projection_coord']
        for i in range(shape[2]):
            hr_coord_d=hr_coord[:,:,i,:].view(-1,3).unsqueeze(0).cuda()
            proj_coord_d=proj_coord[:,:,i,:].view(-1,3).unsqueeze(0).cuda()
            output= model.inference(inp, feat, hr_coord_d, proj_coord_d)
            pred=output['pred'].view(shape[0],shape[1]).cpu().numpy().astype(np.float64)
            sr.append(pred)
        sr=np.stack(sr,axis=-1)
    elif ckpt_name.startswith('MetaSR'):
        inp=torch.FloatTensor(inp).view(1,1,*inp.shape).cuda()
        feat=model.get_feat(inp)
        res=input_matrix_wpn(*shape, scale=spacing_ratio, flatten=False)  
        projection_coord, offset_vector=res['projection_coord'],res['offset_vector']
        for i in range(shape[2]):
            projection_coord_d=projection_coord[:,:,i,:].view(-1,3).unsqueeze(0).cuda()
            offset_vector_d=offset_vector[:,:,i,:].view(-1,6).unsqueeze(0).cuda()
            output=model.inference(feat, projection_coord_d, offset_vector_d)
            pred=output['pred'].view(shape[0],shape[1]).cpu().numpy().astype(np.float64)
            sr.append(pred)
        sr=np.stack(sr,axis=-1)
    else:
        raise Exception("ckpt name is invalid")
    
    sr=np.clip(sr,0,1)   

    if axis==0:
        sr = sr.transpose(2,0,1)
    elif axis==1:
        sr = sr.transpose(0,2,1) 

    return sr