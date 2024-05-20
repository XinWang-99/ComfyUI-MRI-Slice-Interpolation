import numpy as np

import torch


def normalize(image, img_max=0.995, img_min=0.005):


    assert type(image) is torch.Tensor or np.ndarray
    if type(image) is torch.Tensor:
        t_max = torch.quantile(image, img_max)
        t_min = torch.quantile(image, img_min)
    else:
        t_max = np.percentile(image, img_max * 100)
        t_min = np.percentile(image, img_min * 100)
    image = (image - t_min) / (t_max - t_min)
    image[image > 1] = 1
    image[image < 0] = 0
    return image, t_max, t_min