import numpy as np
import torch
import torchvision

from colon3d.util.torch_util import resize_images_batch, to_torch

# --------------------------------------------------------------------------------------------------------------------


def img_to_net_in_format(
    img: np.ndarray | torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    normalize_values: bool = True,
    img_normalize_mean: float = 0.45,
    img_normalize_std: float = 0.225,
    add_batch_dim: bool = False,
    net_in_height: int | None = None,
    net_in_width: int | None = None,
) -> torch.Tensor:
    """Transform an single input image to the network input format.
    Args:
        imgs: the input images [height x width x n_channels] or [height x width]
    Returns:
        imgs: the input images in the network format [1 x n_channels x new_width x new_width] or [1 x new_width x new_width]
    """

    # transform to torch tensor
    img = to_torch(img, device=device).to(dtype)

    # transform to channels first (HWC to CHW format)
    if img.ndim == 3:  # color
        img = torch.permute(img, (2, 0, 1))
    elif img.ndim == 2:  # depth
        img = torch.unsqueeze(img, 0)  # add channel dimension
    else:
        raise ValueError("Invalid image dimension.")


    img = normalize_image_channels(img, img_normalize_mean, img_normalize_std) if normalize_values else img

    if add_batch_dim:
        img = img.unsqueeze(0)

    img = resize_images_batch(
        imgs=img,
        new_height=net_in_height,
        new_width=net_in_width,
    )

    return img


# --------------------------------------------------------------------------------------------------------------------


def normalize_image_channels(img: torch.Tensor, img_normalize_mean: float = 0.45, img_normalize_std: float = 0.225):
    # normalize the values from [0,255] to [0, 1]
    img = img / 255
    # normalize the values to the mean and std of the ImageNet dataset
    img = (img - img_normalize_mean) / img_normalize_std
    return img


# --------------------------------------------------------------------------------------------------------------------


def resize_tensor_image(img: torch.Tensor, new_height: int, new_width: int) -> torch.Tensor:
    """Resize tensor image.
    imgs: the input images [...,height x width]
    new_height: the new height
    new_width: the new width
    """
    # if the image is 2D, add a dimension so Resize can work
    add_dim = False
    if img.ndim == 2:
        add_dim = True
        img = torch.unsqueeze(img, 0)
    img = torchvision.transforms.Resize(size=(new_height, new_width), antialias=True)(img)
    if add_dim:
        img = torch.squeeze(img, 0)
    return img

# --------------------------------------------------------------------------------------------------------------------
