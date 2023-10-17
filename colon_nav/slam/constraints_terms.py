import torch
from torch.nn import functional as nnf

from colon_nav.util.torch_util import get_device

# -------------------------------------------------------------------------------------------------------------------


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


# -------------------------------------------------------------------------------------------------------------------


class SoftConstraints:
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

    # -------------------------------------------------------------------------------------------------------------------

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


# -------------------------------------------------------------------------------------------------------------------
