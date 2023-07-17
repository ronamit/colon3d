import numpy as np
import torch
from torch.nn.functional import normalize

from colon3d.utils.torch_util import assert_2d_tensor, to_default_type

# --------------------------------------------------------------------------------------------------------------------


# @torch.jit.script  # disable this for debugging
def normalize_quaternions(q_in: torch.Tensor) -> torch.Tensor:
    """
    normalize the quaternions to unit quaternions and convert a standard form: one in which the real
    part is non negative.
    Args:
        q_in: Quaternions tensor of shape (..., 4).
    Returns:
        standard unit-quaternions as tensor of shape (..., 4).
    References: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    """
    assert q_in.shape[-1] == 4, "quaternions must be of shape (..., 4)"
    q_out = normalize(q_in, dim=-1, p=2.0, eps=1e-20)
    # change the sign of the quaternion if the real part is negative (to have a standard form)
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
# @torch.jit.script  # disable this for debugging
def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply two unit-quaternions. output = a * b.
    Usual torch rules for broadcasting apply.
    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    References: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    """
    # aw, ax, ay, az = torch.unbind(a, -1)
    # bw, bx, by, bz = torch.unbind(b, -1)
    # ow = aw * bw - ax * bx - ay * by - az * bz
    # ox = aw * bx + ax * bw + ay * bz - az * by
    # oy = aw * by - ax * bz + ay * bw + az * bx
    # oz = aw * bz + ax * by - ay * bx + az * bw
    # ab = torch.stack((ow, ox, oy, oz), -1)

    # change the type to the default type (float64)
    a = to_default_type(a)
    b = to_default_type(b)
    # get the real-part of each quaternion:
    aw = a[..., 0:1]
    bw = b[..., 0:1]
    # get the imaginary part of each quaternion:
    av = a[..., 1:]
    bv = b[..., 1:]
    # compute the real part of the product:
    abw = aw * bw - torch.sum(av * bv, dim=-1, keepdim=True)
    # compute the imaginary part of the product:
    abv = aw * bv + bw * av + torch.cross(av, bv, dim=-1)
    # return the product:
    ab = torch.cat((abw, abv), dim=-1)
    # normalize the result (to avoid numerical errors)
    ab = normalize_quaternions(ab)
    return ab


# --------------------------------------------------------------------------------------------------------------------


def compose_rotations(rot1: torch.Tensor, rot2: torch.Tensor) -> torch.Tensor:
    """composes two rotations  into a single rotation (both are given in the same coordinate system)
        I.e., if R1 and R2 are the rotation matrices corresponding to rot1 and rot2, then we compute R_tot = R2 @ R1
    Args:
        start_rot: [n x 4] unit-quaternions (real part first).
        rot_change:  [n x 4] unit-quaternions (real part first).

    Returns:
        The final rotation [n x 4] unit-quaternions (real part first).
    References:
    """
    # note: the order of multiplication is important (https://math.stackexchange.com/questions/331539/combining-rotation-quaternions
    # the first rotation is the right side in the multiplication
    tot_rot = quaternion_raw_multiply(a=rot2, b=rot1)
    return tot_rot


# --------------------------------------------------------------------------------------------------------------------


def find_rotation_delta(rot1: torch.Tensor, rot2: torch.Tensor) -> torch.Tensor:
    """Find the rotation that needs to be applied after rot1 to get in total the rotation rot2.
        I.e., if R1 and R2 are the rotation matrices corresponding to rot1 and rot2, then we compute R2 = R_delta @ R1
    Args:
        rot1:  rotation as a tensor of shape (..., 4), given as unit-quaternions with real part first.
        rot2:   rotation as a tensor of shape (..., 4),given as unit-quaternions with real part first.
    Returns:
        The rotation delta, a tensor of quaternions of shape (..., 4), given as unit-quaternions with real part first.
    """

    rot_delta = compose_rotations(rot1=invert_rotation(rot1), rot2=rot2)
    return rot_delta


# --------------------------------------------------------------------------------------------------------------------
# @torch.jit.script  # disable this for debugging
def invert_rotation(quaternion: torch.Tensor) -> torch.Tensor:
    """Given a quaternion representing rotation, get the quaternion representing its inverse.
        I.e., if R is the rotation matrix corresponding to a quaternion q, then we compute R.T == R.inv()
    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """
    scaling = torch.tensor([1, -1, -1, -1], device=quaternion.device)
    return quaternion * scaling

# --------------------------------------------------------------------------------------------------------------------


# @torch.jit.script  # disable this for debugging
def rotate_points(points3d: torch.Tensor, rot_vecs: torch.Tensor):
    """Rotate points by given unit-quaternion rotation vectors.
        I.e., if R is the rotation matrix corresponding to a quaternion q, then we compute R @ point
    Args:
        points3d (torch.Tensor): [n_points x 3] each row is  (x, y, z) coordinates of a point to be rotated.
        rot_vecs (torch.Tensor): [n_points x 4] each row is the unit-quaternion of the rotation (q0, qx, qy, qz).
    Returns:
        rotated_points3d (np.array): [n_points x 3] rotated points.
    References:
        https://gamedev.stackexchange.com/questions/28395/rotating-vec tor3-by-a-quaternion
        https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Performance_comparisons
    """
    # change the type of point3d:
    points3d = points3d.to(rot_vecs.dtype)
    points3d = assert_2d_tensor(points3d, 3)
    n_points = points3d.shape[0]
    if rot_vecs.ndim == 1:
        # in case we have a single rotation vector, repeat it for all points
        rot_vecs = rot_vecs.expand(n_points, -1)
    assert_2d_tensor(rot_vecs, 4)

    # # extend the original points with a column of zeros (for the real part of the quaternion)
    # point3d_ext = torch.cat((torch.zeros(n_points, 1, device=points3d.device), points3d), dim=1)  # [n x 4]
    # # rotate the points
    # rot_vecs_inv = invert_rotation(rot_vecs)
    # rotated_points = quaternion_raw_multiply(rot_vecs, point3d_ext)
    # rotated_points = quaternion_raw_multiply(rotated_points, rot_vecs_inv)
    # # take only the last 3 columns (the first column is the real part of the quaternion)
    # rotated_points = rotated_points[:, 1:]

    # w = rot_vecs[:, 0].unsqueeze(-1)  # [n x 1] real part of quaternion qw
    # r = rot_vecs[:, 1:]  # [n x 3] (qx, qy, qz)
    # rotated_points = points3d + 2 * torch.cross(r, w * points3d + torch.cross(r, points3d, dim=1), dim=1)

    s = rot_vecs[:, 0].unsqueeze(-1)  # [n x 1] real part of quaternion qw
    u = rot_vecs[:, 1:]  # [n x 3] (qx, qy, qz)
    rotated_points = (
        2 * u * torch.sum(u * points3d, dim=-1, keepdim=True)
        + points3d * (s**2 - torch.sum(u * u, dim=-1, keepdim=True))
        + 2 * s * torch.cross(u, points3d, dim=1)
    )

    return rotated_points


# --------------------------------------------------------------------------------------------------------------------
# @torch.jit.script  # disable this for debugging
def get_rotation_angles(rot_q: torch.Tensor) -> torch.Tensor:
    """Get the rotation angle of the given rotation quaternions (of the axis-angle representation)
    Args:
        rot_q (torch.Tensor): [n_vecs x 4] each row is a unit-quaternion of the rotation in the format (q0, qx, qy, qz).
    Returns:
        rot_angles (torch.Tensor): [n_vecs] rotation angles in radians. (in the range [-pi, pi])
    Notes:
        *see https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#Unit_quaternions
    """
    assert rot_q.ndim == 2
    assert rot_q.shape[1] == 4
    rot_angles = 2 * torch.atan2(torch.norm(rot_q[:, 1:], dim=1), rot_q[:, 0])
    return rot_angles


# --------------------------------------------------------------------------------------------------------------------


def get_rotation_angle(rot_q: torch.Tensor) -> torch.Tensor:
    """Get the rotation angle of the given rotation quaternion (of the axis-angle representation)
    Args:
        rot_quat (torch.Tensor): [4] unit-quaternion of the rotation in the format (q0, qx, qy, qz).
    Returns:
        rot_angle (torch.Tensor): [1] rotation angle in radians (in the range [-pi, pi])
    """
    assert rot_q.ndim == 1
    assert rot_q.shape[0] == 4
    rot_angle = get_rotation_angles(rot_q.unsqueeze(0))[0]
    return rot_angle


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
