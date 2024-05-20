from .nifti_process import LoadNifti, To_4D_tensor, ImageNormalize, SaveNifti
from .model_loader import ModelLoader
from .slice_interpolation import SliceInterpolation

NODE_CLASS_MAPPINGS = {
    "ImageNormalize": ImageNormalize,
    "LoadNifti": LoadNifti,
    "SaveNifti": SaveNifti,
    "To_4D_tensor": To_4D_tensor,
    "ModelLoader":ModelLoader,
    "SliceInterpolation":SliceInterpolation
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS","WEB_DIRECTORY"]
