import yaml
import folder_paths
import os
import importlib
import torch

config_dir="./custom_nodes/child_node/configs" 
ckpt_dir='./custom_nodes/child_node/ckpts'

class ModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "ckpt_name": (['SAINR','ArSSR','MetaSR','EDSR_x4','EDSR_x5'], ),
                             }}
        
    RETURN_TYPES = ("MODEL","STRING",)
    RETURN_NAMES = ("model","ckpt_name",)
    FUNCTION = "load_model"

    CATEGORY = "child/ModelLoader"

    def load_model(self, ckpt_name):
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
        
        return (model.cuda(),ckpt_name)