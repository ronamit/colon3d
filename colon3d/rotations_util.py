import numpy as np
import torch
from torch.nn.functional import normalize

# --------------------------------------------------------------------------------------------------------------------


@torch.jit.script  # disable this for debugging
def normalize_quaternion(q_in: torch.Tensor) -> torch.Tensor:
    """
    normalize the quaternion to a unit quaternion and convert a standard form: one in which the real
    part is non negative.
    Args:
        q_in: Quaternions tensor of shape (..., 4).
    Returns:
        standard unit-quaternions as tensor of shape (..., 4).
    References: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    """
    assert q_in.shape[-1] == 4, "quaternions must be of shape (..., 4)"
    q_out = normalize(q_in, dim=-1, p=2.0, eps=1e-18)
    q_out = torch.where(q_out[..., 0:1] < 0, -q_out, q_out)
    return q_out


# --------------------------------------------------------------------------------------------------------------------


def get_identity_quaternion() -> torch.Tensor:
    """The identity quaternion (in real part first format)
    Returns:
        np.ndarray:  the identity quaternion. It represents no rotation.
    """
    return torch.tensor([1.0, 0.0, 0.0, 0.0])


# --------------------------------------------------------------------------------------------------------------------
@torch.jit.script  # disable this for debugging
def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions.  Usual torch rules for broadcasting apply.
    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    References: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


# --------------------------------------------------------------------------------------------------------------------


def apply_rotation_change(start_rot: torch.Tensor, rot_change: torch.Tensor) -> torch.Tensor:
    """apply a rotation to a rotation (both are given in the same coordinate system)
    Args:
        start_rot: [n x 4] unit-quaternions (real part first).
        rot_change:  [n x 4] unit-quaternions (real part first).

    Returns:
        The final rotation [n x 4] unit-quaternions (real part first).
    References:
    """
    # note: the order of multiplication is important (https://math.stackexchange.com/questions/331539/combining-rotation-quaternions
    # the first rotation is the right side in the multiplication
    final_rot = quaternion_raw_multiply(a=rot_change, b=start_rot)
    return final_rot


# --------------------------------------------------------------------------------------------------------------------


def find_rotation_change(start_rot: torch.Tensor, final_rot: torch.Tensor) -> torch.Tensor:
    """Find the rotation change from one rotation to another.
    Args:
        start_rot: Initial rotation, as a tensor of shape (..., 4), real part first.
        final_rot: Final rotation, as a tensor of shape (..., 4), real part first.
    Returns:
        The rotation change, a tensor of quaternions of shape (..., 4)."""
        
    rot_change = quaternion_raw_multiply(a=final_rot, b=invert_rotation(start_rot))
    return rot_change


# --------------------------------------------------------------------------------------------------------------------
@torch.jit.script  # disable this for debugging
def invert_rotation(quaternion: torch.Tensor) -> torch.Tensor:
    """Given a quaternion representing rotation, get the quaternion representing its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    scaling = torch.tensor([1, -1, -1, -1], device=quaternion.device)
    return quaternion * scaling


# --------------------------------------------------------------------------------------------------------------------


@torch.jit.script  # disable this for debugging
def quaternion_apply(quaternion: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """
    Apply the rotation given by a quaternion to a 3D point.
    Usual torch rules for broadcasting apply.
    Note that we use "passive rotation", i.e. the rotation is applied to the coordinate system.
    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4).
        point: Tensor of 3D points of shape (..., 3).
    Returns:
        Tensor of rotated points of shape (..., 3).
    References:
                https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
                https://gamedev.stackexchange.com/questions/28395/rotating-vector3-by-a-quaternion
    """
    assert point.size(-1) == 3, f"Points are not in 3D, {point.shape}."
    assert quaternion.size(-1) == 4, f"Quaternions are not in 4D, {quaternion.shape}."
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        invert_rotation(quaternion),
    )
    return out[..., 1:]


# --------------------------------------------------------------------------------------------------------------------


def rotate(points3d: torch.Tensor, rot_vecs: torch.Tensor):
    """Rotate points by given unit-quaternion rotation vectors.
        Note that we use "passive rotation", i.e. the rotation is applied to the coordinate system.
    Args:
        points3d (torch.Tensor): [n_points x 3] each row is  (x, y, z) coordinates of a point to be rotated.
        rot_vecs (torch.Tensor): [n_points x 4] each row is the unit-quaternion of the rotation (q0, qx, qy, qz).
    Returns:
        rotated_points3d (np.array): [n_points x 3] rotated points.
    References:
        https://gamedev.stackexchange.com/questions/28395/rotating-vector3-by-a-quaternion
    """
    return quaternion_apply(quaternion=rot_vecs, point=points3d)


# --------------------------------------------------------------------------------------------------------------------


@torch.jit.script  # disable this for debugging
def get_cos_half_angle_between_rotations(rot1: torch.Tensor, rot2: torch.Tensor):
    """
    Find the smallest angle to rotate a unit-quaternion rot2 to become rot1.
    see https://math.stackexchange.com/questions/3999557/how-to-compare-two-rotations-represented-by-axis-angle-rotation-vectors
    Args:
        @rot1 (torch.Tensor)[vector of size 4]:  a unit-quaternion of the rotation in the format (q0, qx, qy, qz).
        rot2 (torch.Tensor)[vector of size 4]:  a unit-quaternion of the rotation in the format (q0, qx, qy, qz).
    Returns:
        cos_half_angle_change (torch.Tensor) [scalar] [rad]: cosine of half the  smallest angle to rotate a unit-quaternion rot1 to become rot2 (units: rad)
    """
    assert rot1.shape == rot2.shape == (4,)
    # find the quaternion representing the rotation from rot2 to rot1 by dividing rot1 by rot2
    inv_rot2 = invert_rotation(rot2.unsqueeze(0))
    quotient = quaternion_raw_multiply(rot1.unsqueeze(0), inv_rot2)
    # find the angle of the rotation
    real_part = torch.abs(quotient[:, 0])
    # find the cosine of half the angle of the rotation
    cos_half_angle_change = real_part
    cos_half_angle_change = torch.clamp(cos_half_angle_change, -1.0, 1.0)
    return cos_half_angle_change.squeeze()

# --------------------------------------------------------------------------------------------------------------------


def get_smallest_angle_between_rotations(rot1: torch.Tensor, rot2: torch.Tensor):
    cos_half_angle_change = get_cos_half_angle_between_rotations(rot1, rot2)
    angle = 2 * torch.acos(cos_half_angle_change)
    return angle

# --------------------------------------------------------------------------------------------------------------------


def get_random_rot_quat(rng: np.random.Generator, angle_std_deg: float, n_vecs: int):
    """Get a random rotation quaternion.
    Args:
        rng (np.random.Generator): random number generator
        angle_std_deg (float): standard deviation of the rotation angle (in degrees)
        n_vecs (int): number of random rotation vectors to generate
    Returns:
        rot_quat (np.ndarray): [n_vecs x 4] each row is a unit-quaternion of the rotation in the format (q0, qx, qy, qz).
    """
    angle_std_rad = np.deg2rad(angle_std_deg)
    angle_err = rng.standard_normal(size=(n_vecs, 1)) * angle_std_rad
    err_dir = rng.random(size=(n_vecs, 3))
    err_dir /= np.linalg.norm(err_dir, axis=1, keepdims=True)
    rot_quat = np.concatenate([np.cos(angle_err / 2), np.sin(angle_err / 2) * err_dir], axis=1)
    return rot_quat


# ----------------------------------------------------------------------
