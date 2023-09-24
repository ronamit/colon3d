import numpy as np
import PIL
import torch
import torchvision

# --------------------------------------------------------------------------------------------------------------------


def mps_available():
    # Mac ARM
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


# --------------------------------------------------------------------------------------------------------------------


def get_device(gpu_id: int = 0):
    if torch.cuda.is_available():
        return torch.device("cuda", gpu_id)
    if mps_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------------------------------------------------


def get_default_dtype(package="torch", num_type=None):
    num_type = num_type or "float"
    if package == "torch":
        if num_type in ["float", "float_m"]:
            return torch.float32 if mps_available() else torch.float64
        if num_type == "float_m":
            # the precision for depth maps
            return torch.float32
        if num_type == "int":
            return torch.int32
        raise ValueError(f"Unknown num_type: {num_type}")
    if package == "numpy":
        if num_type in ["float", "float_m"]:
            return np.float64
        if num_type == "float_m":
            # the precision for depth maps
            return np.float32
        if num_type == "int":
            return np.int64
        raise ValueError(f"Unknown num_type: {num_type}")
    raise ValueError(f"Unknown package: {package}")


# --------------------------------------------------------------------------------------------------------------------


def to_default_type(x, num_type="float"):
    if isinstance(x, torch.Tensor):
        return to_torch(x, num_type=num_type)
    if isinstance(x, np.ndarray):
        return to_numpy(x, num_type=num_type)
    return x


# --------------------------------------------------------------------------------------------------------------------


def to_numpy(x, num_type=None, dtype=None):
    if dtype is None:
        dtype = get_default_dtype("numpy", num_type)
    if isinstance(x, np.ndarray):
        return x.astype(dtype)
    if isinstance(x, dict):
        return {k: to_numpy(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_numpy(v) for v in x]
    if isinstance(x, torch.Tensor):
        return x.numpy(force=True).astype(dtype)
    if isinstance(x, PIL.Image.Image):
        return np.array(x).astype(dtype)
    return x


# --------------------------------------------------------------------------------------------------------------------


def to_torch(x, num_type=None, dtype=None, device=None):
    if dtype is None:
        dtype = get_default_dtype("torch", num_type)
    device = device or get_device()
    if isinstance(x, torch.Tensor):
        return x.to(dtype).to(device)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype).to(device)
    if isinstance(x, PIL.Image.Image):
        return torch.from_numpy(np.array(x)).to(dtype).to(device)
    if isinstance(x, dict):
        return {k: to_torch(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_torch(v) for v in x]
    return x  # all other types - do nothing


# --------------------------------------------------------------------------------------------------------------------


def concat_list_to_tensor(x, num_type=None, device=None):
    # to prevent warning from torch - first convert to numpy array
    return to_torch(np.array(x), num_type, device)


# --------------------------------------------------------------------------------------------------------------------


def np_func(func):
    """Decorator that chttps://beta.ruff.rs/docs/rules/unnecessary-comprehension-any-allonverts all Numpy arrays to PyTorch tensors before calling the function and converts the result back to Numpy array."""

    def wrapper(*args, **kwargs):
        args = [to_torch(x) for x in args]
        kwargs = {k: to_torch(v) for k, v in kwargs.items()}
        result = func(*args, **kwargs)
        return to_numpy(result)

    return wrapper


# --------------------------------------------------------------------------------------------------------------------


def get_val(x):
    if isinstance(x, torch.Tensor):
        return x.item()
    return x


# --------------------------------------------------------------------------------------------------------------------


def assert_2d_tensor(t: torch.Tensor, dim2: int):
    if t.ndim == 1:
        t = t.unsqueeze(0)  # add the "number of samples" dimension
    assert t.ndim == 2, f"Tensor should be [n x {dim2}]."
    assert t.shape[1] == dim2, f"Tensor should be [n x {dim2}]."
    assert is_finite(t), "Tensor should be finite."
    return t


# --------------------------------------------------------------------------------------------------------------------


def assert_2d_array(t: np.ndarray, dim2: int):
    if t.ndim == 1:
        t = t[np.newaxis, :]  # add the "number of samples" dimension
    assert t.ndim == 2, f"Tensor should be [n x {dim2}]."
    assert t.shape[1] == dim2, f"Tensor should be [n x {dim2}]."
    assert is_finite(t), "Tensor should be finite."
    return t


# --------------------------------------------------------------------------------------------------------------------


def assert_1d_tensor(t: torch.Tensor):
    if t.ndim == 0:
        t = t.unsqueeze(0)  # add the "number of samples" dimension
    assert t.ndim == 1, "Tensor should be 1D."
    assert is_finite(t), "Tensor should be finite."
    return t


# --------------------------------------------------------------------------------------------------------------------
def assert_same_sample_num(tensor_list: tuple):
    n = tensor_list[0].shape[0]
    for t in tensor_list:
        assert t.shape[0] == n, "Different number of samples."
    return n


# --------------------------------------------------------------------------------------------------------------------


def is_finite(x):
    if isinstance(x, torch.Tensor):
        return torch.all(torch.isfinite(x))
    if isinstance(x, np.ndarray):
        return np.all(np.isfinite(x))
    return np.isfinite(x)


# --------------------------------------------------------------------------------------------------------------------


def pseudo_huber_loss_on_x_sqr(x_sqr, delta=1.0):
    """Pseudo-Huber loss function.
    References
        https://en.wikipedia.org/wiki/Huber_loss
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.pseudo_huber.html
    """
    losses = delta**2 * (torch.sqrt(1 + x_sqr / delta**2) - 1)
    return losses


# --------------------------------------------------------------------------------------------------------------------


def resize_single_image(
    img: torch.Tensor,
    new_height: int,
    new_width: int,
) -> torch.Tensor:
    """Resize an image that is in a torch tensor format.
    img: single input image [height x width] or [n_channels x height x width]
    new_height: the new height
    new_width: the new width
    """
    img = img.unsqueeze(0)  # add batch dimension
    resizer = torchvision.transforms.Resize((new_height, new_width), antialias=True)
    resized_img = resizer(img)
    resized_img = resized_img.squeeze(0)  # remove batch dimension
    return resized_img


# --------------------------------------------------------------------------------------------------------------------


def resize_images_batch(
    imgs: torch.Tensor,
    new_height: int,
    new_width: int,
) -> torch.Tensor:
    """Resize a batch of images that are in a torch tensor format.
    imgs: input images [batch_size x n_channels x height x width]
    new_height: the new height
    new_width: the new width
    """
    resizer = torchvision.transforms.Resize((new_height, new_width), antialias=True)
    resized_imgs = resizer(imgs)
    return resized_imgs


# --------------------------------------------------------------------------------------------------------------------
