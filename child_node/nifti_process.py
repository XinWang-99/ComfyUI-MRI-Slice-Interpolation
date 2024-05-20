
import hashlib
import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image, ImageOps, ImageSequence
from .utils.image import normalize
import folder_paths
from comfy.cli_args import args
import os

# input_dir="./input/nifti"
input_dir = os.path.join(folder_paths.get_input_directory(),'nifti')
output_dir = os.path.join(folder_paths.get_output_directory(),'nifti')

if not os.path.exists(input_dir):
    os.makedirs(input_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class LoadNifti:

    @classmethod
    def INPUT_TYPES(s):
        # input_dir = folder_paths.get_input_directory()
        
        # print("input_dir",input_dir)
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        # print("files",files)
        return {
            "required": {
                "nifti_file": (sorted(files),),
            },
        }

    RETURN_TYPES =("IMAGE","SOURCE","STRING")
    RETURN_NAMES = ("IMAGE","SOURCE","FILENAME")
    CATEGORY = "child/Nifti"
    FUNCTION = "execute"

    def execute(self, nifti_file):
        # nifti_path=folder_paths.get_annotated_filepath(nifti_file)
        nifti_path=os.path.join(input_dir,nifti_file)
        print("nifti_file",nifti_file)
        data = sitk.ReadImage(nifti_path)
        image = sitk.GetArrayFromImage(data).astype(np.float32)
        print(image.shape,data.GetSpacing())

        return (image,data,nifti_file)


class SaveNifti:
   
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"image": ("IMAGE",),
                     "source": ("SOURCE",),
                     "ratio": ("INT",{"default": 2}),
                     "filename": ("STRING",{"default": "none"}),
                     "filename_prefix": ("STRING", {"default": "SR"})}
                }

    RETURN_TYPES = ("STRING",)
    # RETURN_NAMES=("OUTPUT_PATH",)
    FUNCTION = "execute"

    # OUTPUT_NODE = True

    CATEGORY = "child/Nifti"

    def execute(self, image, source, ratio, filename,filename_prefix):
        # print(111,images.shape)

        def renormalize(ori,img):
            img=img*(ori.max()-ori.min())+ori.min()
            return img.astype(np.int16)
        
        ori_img=sitk.GetArrayFromImage(source)
        image=renormalize(ori_img,image)

        Image=sitk.GetImageFromArray(image)
        Image.SetOrigin(source.GetOrigin())
        Image.SetDirection(source.GetDirection())
        spacing=source.GetSpacing()
        Image.SetSpacing((spacing[0],spacing[1],spacing[2]/ratio))

        output_path=os.path.join(output_dir,filename_prefix+'_'+filename+'.nii.gz')
        sitk.WriteImage(Image,output_path)

        return (output_path,)
        
   

       
    
class To_4D_tensor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "child/Nifti"

    def execute(self, image):
        assert type(image) is torch.Tensor or np.ndarray
        if type(image) is torch.Tensor:
            if len(image.shape) == 3:
                image = image.unsqueeze(-1).repeat(1, 1, 1, 3)
        else:
            if image.ndim == 3:
                image = np.tile(image[:, :, :, np.newaxis], 3)
                image = torch.from_numpy(image)
        return (image,)

class ImageNormalize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "max_ratio": (
                    "FLOAT",
                    {
                        "default": 99.5,
                        "min": 0,  # Minimum value
                        "max": 100,  # Maximum value
                        "step": 1,  # Slider's step
                        "display": "number",  # Cosmetic only: display as "number" or "slider"
                    },
                ),
                "min_ratio": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0,  # Minimum value
                        "max": 100,  # Maximum value
                        "step": 1,  # Slider's step
                        "display": "number",  # Cosmetic only: display as "number" or "slider"
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_normailze"
    CATEGORY = "child/ImageProcessing"

    def image_normailze(self, image, max_ratio, min_ratio):
        image, t_max, t_min = normalize(image, max_ratio / 100, min_ratio / 100)
        return (image,)
    

