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
    """Get the default data type for a package.
    Args:
        package: the package name, either "torch" or "numpy"
        num_type: the numeric type, either "float" or "int" or "float_m".
    """
    num_type = num_type or "float"
    if package == "torch":
        if num_type == "float":
            return torch.float32
        if num_type == "float_m":
            return torch.float32
        if num_type == "int":
            return torch.int32
        raise ValueError(f"Unknown num_type: {num_type}")
    if package == "numpy":
        if num_type == "float":
            return np.float32
        if num_type == "float_m":
            return np.float32
        if num_type == "int":
            return np.int32
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
        return x.detach().numpy(force=True).astype(dtype)
    if isinstance(x, PIL.Image.Image):
        return np.array(x).astype(dtype)
    return x


# --------------------------------------------------------------------------------------------------------------------


def to_device(x: torch.Tensor, device: torch.device):
    """Move a tensor to a device.
    Sources:
        * https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/3
    """
    return x.to(device, non_blocking=True)


# --------------------------------------------------------------------------------------------------------------------


def to_torch(x, num_type=None, dtype=None, device: None | torch.device | str = None):
    """Covert various types to torch tensors."""
    if dtype is None:
        dtype = get_default_dtype("torch", num_type)
    # If dict or list - Recursively convert to torch tensors and keep the same structure.
    if isinstance(x, dict):
        return {k: to_torch(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_torch(v) for v in x]
    # If torch tensor - Convert to the desired type.
    if isinstance(x, torch.Tensor):
        x = x.to(dtype)
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(dtype)
    if isinstance(x, PIL.Image.Image):
        x = torch.from_numpy(np.array(x)).to(dtype)
    if isinstance(x, int | float):
        x = torch.tensor(x).to(dtype)
    # If device is specified - Move to the device.
    if device == "default":
        device = get_device()
    if device is not None:
        x = to_device(x, device)
    return x


# ---------------------------------------------------------------------------------------------------------------------


def sample_to_gpu(sample: dict, device: torch.device | None = None) -> dict:
    """
    Note: this must be applied only after a sampled batch is created by the data loader.
    See: https://github.com/pytorch/pytorch/issues/98002#issuecomment-1511972876
    """
    if device is None:
        device = get_device()
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            sample[k] = to_device(v, device=device)
    return sample


# --------------------------------------------------------------------------------------------------------------------


def concat_list_to_tensor(x, num_type=None, device=None):
    # to prevent warning from torch - first convert to numpy array
    return to_torch(np.array(x), num_type=num_type, device=device)


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
    assert t.ndim == 2, f"Tensor should be [n x {dim2}]."
    assert t.shape[1] == dim2, f"Tensor should be [n x {dim2}]."
    assert is_finite(t), "Tensor should be finite."
    return t


# --------------------------------------------------------------------------------------------------------------------


def assert_2d_array(t: np.ndarray, dim2: int):
    assert t.ndim == 2, f"Tensor should be [n x {dim2}]."
    assert t.shape[1] == dim2, f"Tensor should be [n x {dim2}]."
    assert is_finite(t), "Tensor should be finite."
    return t


# --------------------------------------------------------------------------------------------------------------------


def assert_1d_tensor(t: torch.Tensor):
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
