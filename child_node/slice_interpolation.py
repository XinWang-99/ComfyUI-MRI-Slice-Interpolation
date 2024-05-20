import os
import torch
import folder_paths
from .super_resolution.test import test
import yaml
import importlib
import torch
from torch.cuda.amp import autocast as autocast

config_dir="./custom_nodes/child_node/configs" 
ckpt_dir='./custom_nodes/child_node/ckpts'
input_dir = os.path.join(folder_paths.get_input_directory(),'nifti')
# class SliceInterpolation:
#     @classmethod
#     def INPUT_TYPES(s):

#         files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

#         return {"required": { 
#             "model":("MODEL",),
#             "image": (sorted(files),),
#             "spacing": ("FLOAT", {"default": 1, "min": 0.1, "max": 10, "step": 0.1}),
#              "axis": ("INT",{"default": 2, "min": 0, "max": 2, "step": 1})}}
                         
#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "generate"

#     CATEGORY = "child/SliceInterpolation"

#     def generate(self,model,nifti_file,spacing,axis):
#         nifti_path=os.path.join(input_dir,nifti_file)
#         print(nifti_path)
#         with torch.no_grad():
#             sr = test(model,nifti_path,spacing,axis)
#         return (sr,)

class SliceInterpolation:
    @classmethod
    def INPUT_TYPES(s):

        return {"required": { 
           "ckpt_name": (['SAINR','ArSSR','MetaSR','EDSR_x4','EDSR_x5'],),
            "image": ("IMAGE",),
            "ratio": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
             "axis": ("INT",{"default": 0, "min": 0, "max": 2, "step": 1})}}
                         
    RETURN_TYPES = ("IMAGE","INT")
    RETURN_NAMES=("image","ratio")
    FUNCTION = "generate"

    CATEGORY = "child/SliceInterpolation"

    def generate(self,ckpt_name,image,ratio,axis):
        def load_model(ckpt_name):
            config_path=os.path.join(config_dir,ckpt_name+'.yaml')
            with open(config_path, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            print('config loaded.')
            model_config=config.get('model')
            module, cls = model_config['name'].rsplit(".", 1)
    
            model=getattr(importlib.import_module(module, package=None), cls)(**model_config['args'])

            ckpt_path=os.path.join(ckpt_dir,ckpt_name+'.pth')
            st=torch.load(ckpt_path,map_location='cpu')['model']
            model.load_state_dict(st)
            return model.cuda()
        
        model=load_model(ckpt_name)
        with torch.no_grad():
            with autocast():
                sr = test(ckpt_name,model,image,ratio,axis)
        return (sr,ratio)
    
    