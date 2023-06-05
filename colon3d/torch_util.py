import numpy as np
import torch
from torch.nn import functional as nnf

# --------------------------------------------------------------------------------------------------------------------


def get_device():
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_available:
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------------------------------------------------


def to_numpy(x, dtype=np.float64):
    if isinstance(x, torch.Tensor):
        return x.numpy(force=True).astype(dtype)
    return x


# --------------------------------------------------------------------------------------------------------------------


def np_func(func):
    """Decorator that converts all Numpy arrays to PyTorch tensors before calling the function and converts the result back to Numpy array."""

    def wrapper(*args, **kwargs):
        args = [torch.from_numpy(x) if isinstance(x, np.ndarray) else x for x in args]
        kwargs = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in kwargs.items()}
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


# --------------------------------------------------------------------------------------------------------------------


def assert_1d_tensor(t: torch.Tensor):
    assert t.ndim == 1, "Tensor should be 1D."

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


class SoftConstraints:
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(
        self,
        constraints_dict,
        default_barrier_jump: float = 20.0,
        default_margin_ratio=0.1,
        default_weight: float = 1.0,
    ):
        self.device = get_device()
        self.constraints = {}
        for constraint_name, constraint_params in constraints_dict.items():
            barrier_jump = torch.tensor(constraint_params.get("barrier_jump", default_barrier_jump), device=self.device)
            margin_ratio = torch.tensor(constraint_params.get("margin_ratio", default_margin_ratio), device=self.device)
            eps = torch.exp(-barrier_jump)
            lim = torch.tensor(constraint_params["lim"], device=self.device)
            margin = margin_ratio * torch.abs(lim)
            weight = torch.tensor(constraint_params.get("weight", default_weight), device=self.device)
            self.constraints[constraint_name] = {
                "lim_type": constraint_params["lim_type"],
                "lim": lim,
                "barrier_jump": barrier_jump,
                "eps": eps,
                "weight": weight,
                "margin": margin,
            }

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_penalty(self, constraint_name: str, val: torch.Tensor) -> torch.Tensor:
        """
        Log barrier penalty function for enforcing a constraint on a scalar value.
        For soft upper limit:
            We define margin = |margin_ratio * upper_lim|
            when val < (upper_lim - margin), the penalty is 0
            when  (upper_lim - margin) < val < upper_lim , the penalty increasing sharply
            when val > margin , the penalty jumps by barrier_jump and then increases linearly
        For soft lower limit:
            We define margin = |margin_ratio * lower_lim|
            when val > (lower_lim + margin), the penalty is 0
            when  val is deceasing in the range (lower_lim, lower_lim + margin), the penalty increasing sharply
            when val decreases even further beneath  lower_lim, the penalty jumps by barrier_jump and then increases linearly

        Note: after violating the constraint, we make the gradient non-zero, otherwise the optimizer will not move

        """
        constraint_param = self.constraints[constraint_name]
        lim_type = constraint_param["lim_type"]
        lim = constraint_param["lim"]
        weight = constraint_param["weight"]
        barrier_jump = constraint_param["barrier_jump"]
        margin = constraint_param["margin"]
        eps = constraint_param["eps"]
        if lim_type == "upper":
            penalty = upper_log_barrier(val, lim, barrier_jump=barrier_jump, eps=eps, margin=margin)
        elif lim_type == "lower":
            penalty = lower_log_barrier(val, lim, barrier_jump=barrier_jump, eps=eps, margin=margin)
        else:
            raise ValueError(f"Invalid lim_type {lim_type}")
        return penalty * weight

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# --------------------------------------------------------------------------------------------------------------------


@torch.jit.script  # disable this for debugging
def upper_log_barrier(
    val: torch.Tensor,
    upper_lim: torch.Tensor,
    barrier_jump: torch.Tensor,
    eps: torch.Tensor,
    margin: torch.Tensor,
) -> torch.Tensor:
    """
    when val < (upper_lim - margin), the penalty is 0
    when  (upper_lim - margin) < val < upper_lim , the penalty increasing sharply
    when val > margin , the penalty jumps by barrier_jump and then increases linearly
    """
    if val < (upper_lim - margin):
        return torch.zeros_like(val)
    if (upper_lim - margin) < val < upper_lim:
        return nnf.relu(-torch.log(nnf.relu(upper_lim - val) + eps))
    return barrier_jump + val - upper_lim


# -------------------------------------------------------------------------------------------------------------------


@torch.jit.script  # disable this for debugging
def lower_log_barrier(
    val: torch.Tensor,
    lower_lim: torch.Tensor,
    barrier_jump: torch.Tensor,
    eps: torch.Tensor,
    margin: torch.Tensor,
) -> torch.Tensor:
    """
    when val > (lower_lim + margin), the penalty is 0
    when  val is deceasing in the range (lower_lim, lower_lim + margin), the penalty increasing sharply
    when val decreases even further beneath  lower_lim, the penalty jumps by barrier_jump and then increases linearly
    """
    if val > lower_lim + margin:
        return torch.zeros_like(val)
    if lower_lim < val < (lower_lim + margin):
        return nnf.relu(-torch.log(nnf.relu(val - lower_lim) + eps))
    return barrier_jump + lower_lim - val


# --------------------------------------------------------------------------------------------------------------------
