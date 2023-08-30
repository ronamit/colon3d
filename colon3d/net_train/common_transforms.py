import cv2
import numpy as np
import torch

from colon3d.util.torch_util import to_numpy, to_torch

# --------------------------------------------------------------------------------------------------------------------


class AllToNumpy:
    def __call__(self, sample: dict) -> dict:
        for k, v in sample.items():
            sample[k] = to_numpy(v, dtype=np.float32)
        return sample


# --------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------
def resize_image(img: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
    """Resize an image of shape [height x width] or [height x width x n_channels]"""
    if img.ndim == 1:  # depth
        return cv2.resize(img, dsize=(new_width, new_height), fx=1.0, fy=1.0, interpolation=cv2.INTER_NEAREST)
    if img.ndim == 3:  # color
        return cv2.resize(img, dsize=(new_width, new_height), interpolation=cv2.INTER_LINEAR)
    raise ValueError("Invalid image dimension.")


# --------------------------------------------------------------------------------------------------------------------


def img_to_net_in_format(
    img: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
    normalize_values: bool = True,
    img_normalize_mean: float = 0.45,
    img_normalize_std: float = 0.225,
    new_height: int | None = None,
    new_width: int | None = None,
    add_batch_dim: bool = False,
) -> torch.Tensor:
    """Transform an single input image to the network input format.
    Args:
        imgs: the input images [height x width x n_channels] or [height x width]
    Returns:
        imgs: the input images in the network format [1 x n_channels x new_width x new_width] or [1 x new_width x new_width]
    """
    height, width = img.shape[:2]

    if new_height and new_width and (height, width) != (new_height, new_width):
        # resize the images
        img = resize_image(img, new_height=new_height, new_width=new_width)

    # transform to channels first (HWC to CHW format)
    if img.ndim == 3:  # color
        img = np.transpose(img, (2, 0, 1))
    elif img.ndim == 2:  # depth
        img = np.expand_dims(img, axis=0)
    else:
        raise ValueError("Invalid image dimension.")

    # transform to torch tensor and normalize the values to [0, 1]
    img = to_torch(img, device=device).to(dtype)

    if normalize_values:
        # normalize the values from [0,255] to [0, 1]
        img = img / 255
        # normalize the values to the mean and std of the ImageNet dataset
        img = (img - img_normalize_mean) / img_normalize_std
    if add_batch_dim:
        img = img.unsqueeze(0)
    return img


# --------------------------------------------------------------------------------------------------------------------
