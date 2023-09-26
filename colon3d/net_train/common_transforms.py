import torch
import torchvision

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
