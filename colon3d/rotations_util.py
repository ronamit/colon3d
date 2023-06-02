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


def get_identity_quaternion_np() -> np.ndarray:
    """The identity quaternion (in real part first format)
    Returns:
        np.ndarray:  the identity quaternion. It represents no rotation.
    """
    return np.array([1.0, 0.0, 0.0, 0.0])


# --------------------------------------------------------------------------------------------------------------------


def get_identity_quaternion() -> torch.Tensor:
    """The identity quaternion (in real part first format)
    Returns:
        np.ndarray:  the identity quaternion. It represents no rotation.
    """
    return torch.as_tensor(get_identity_quaternion_np())


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
def quaternion_raw_multiply_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return quaternion_raw_multiply(torch.as_tensor(a), torch.as_tensor(b)).numpy()


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


def invert_rotation_np(quaternion: np.ndarray) -> np.ndarray:
    """Given a quaternion representing rotation, get the quaternion representing its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """
    inv_quaternion = quaternion.copy()
    inv_quaternion[:, 1:] *= -1
    return inv_quaternion


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


def rotate_np(points3d: np.array, rot_vecs: np.array):
    """Rotate points by given unit-quaternion rotation vectors.
        Note that we use "passive rotation", i.e. the rotation is applied to the coordinate system.
    Args:
        points3d (np.array): [n_points x 3] each row is  (x, y, z) coordinates of a point to be rotated.
        rot_vecs (np.array): [n_points x 4] each row is the unit-quaternion of the rotation (q0, qx, qy, qz).
    Returns:
        rotated_points3d (np.array): [n_points x 3] rotated points.
    """
    rotated_points3d = rotate(torch.from_numpy(points3d), torch.from_numpy(rot_vecs))
    return rotated_points3d.numpy()


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


def get_smallest_angle_between_rotations_np(rot1: np.array, rot2: np.array):
    rot1 = torch.from_numpy(rot1)
    rot2 = torch.from_numpy(rot2)
    cos_half_angle_change = get_cos_half_angle_between_rotations(rot1, rot2)
    cos_half_angle_change_np = cos_half_angle_change.numpy()
    angle = 2 * np.arccos(cos_half_angle_change_np)
    return angle.item()


# --------------------------------------------------------------------------------------------------------------------
def apply_egomotions(prev_poses: torch.Tensor, egomotions: torch.Tensor):
    """Change the camera poses by applying the 6DOF egomotions
    Args:
        poses of the previous time step (torch.Tensor): [n_cam x 7] each row is the camera pose in world system (x, y, z, q0, qx, qy, qz).
        egomotions (torch.Tensor): [n_cam x 7] each row is the camera egomotion (x, y, z, q0, qx, qy, qz).
                    The egomotion is the 6-DOF current camera pose w.r.t. the previous camera pose.
    Returns:
        new_poses (torch.Tensor): [n_cam x 7] each row is the camera pose of the current time step in world system (x, y, z, q0, qx, qy, qz).
    """
    loc_change = egomotions[:, :3]  # the location of the current camera in the previous camera axes
    rot_change = egomotions[:, 3:]  # unit-quaternion of the of the current camera axes w.r.t. the previous camera axes
    prev_loc = prev_poses[:, :3]  # previous cam location in world system
    prev_rot = prev_poses[:, 3:]  # previous rotations as unit-quaternions in world system
    trans_in_world = rotate(loc_change, prev_rot)  # rotate the translation vector to the world system
    new_loc = prev_loc + trans_in_world
    # transform the rotation to the world system
    # combine the rotations, note that the order of multiplication is important..
    # to find the rotation of the current camera w.r.t the world system we first rotate to the previous camera axes, and then apply the rotation of the current cam w.r.t the previous cam
    new_rot = quaternion_raw_multiply(rot_change, prev_rot)
    new_poses = torch.cat((new_loc, new_rot), dim=1)
    return new_poses


# --------------------------------------------------------------------------------------------------------------------


def apply_egomotions_np(poses: np.array, egomotions: np.array):
    """Change the camera poses by applying the 6DOF egomotions
    Args:
        poses (np.ndarray): [n_cam x 7] each row is the camera pose (x, y, z, q0, qx, qy, qz).
        egomotions (np.ndarray): [n_cam x 7] each row is the camera egomotion (pose change) (x, y, z, q0, qx, qy, qz).
    """
    new_poses = apply_egomotions(torch.from_numpy(poses), torch.from_numpy(egomotions))
    return new_poses.numpy()


# --------------------------------------------------------------------------------------------------------------------

def transform_between_poses(poses1: torch.Tensor, poses2: torch.Tensor):
    """Find the transformation from poses1 to poses2.
        Args:
        poses1, poses2 (torch.Tensor): [N X 7] (x, y, z, q0, qx, qy, qz) each row is the camera pose in world system.
    """
    locs1 = poses1[:, :3]
    locs2 = poses2[:, :3]
    rots1 = poses1[:, 3:]
    rots2 = poses2[:, 3:]
    # find the location change in the world system:
    loc_changes = locs2 - locs1
    # rotate the location change to poses1:
    loc_change = rotate(points3d=loc_changes, rot_vecs=invert_rotation(rots1))
    # find the rotation change in the previous camera axes:
    rot_change = quaternion_raw_multiply(rots2, invert_rotation(rots1))
    poses_change = torch.cat((loc_change, rot_change), dim=1)
    return poses_change
# --------------------------------------------------------------------------------------------------------------------

def transform_between_poses_np(poses1: np.array, poses2: np.array):
    """Find the transformation from poses1 to poses2.
        Args:
        poses1, poses2 (np.ndarray): [N X 7] (x, y, z, q0, qx, qy, qz) each row is the camera pose in world
    """
    poses_change = transform_between_poses(torch.from_numpy(poses1), torch.from_numpy(poses2))
    return poses_change.numpy()
# --------------------------------------------------------------------------------------------------------------------


def infer_egomotions(cam_poses: torch.Tensor):
    """infer the egomotions from the previous and current camera poses
    Args:
        cam_poses (torch.Tensor): [n_frames x 7] each row is the camera pose (x, y, z, q0, qx, qy, qz).
    Returns:
        egomotions (torch.Tensor): [n_frames x 7] each row is the camera egomotion (pose change) (x, y, z, q0, qx, qy, qz).
    """
    prev_poses = cam_poses[:-1, :]
    cur_poses = cam_poses[1:, :]
    
    poses_changes = transform_between_poses(poses1=prev_poses, poses2=cur_poses)
    # set the first egomotion to be the identity rotation and zero translation:
    egomotions = torch.zeros_like(cam_poses)
    egomotions[0, :3] = 0
    egomotions[0, 3:] = get_identity_quaternion()
    # set the rest of the egomotions:
    egomotions[1:, :] = poses_changes
    return egomotions

# ----------------------------------------------------------------------


def infer_egomotions_np(cam_poses: np.ndarray):
    """infer the egomotions from the previous and current camera poses
    Args:
        cam_poses (np.ndarray): [n_frames x 7] each row is the camera pose (x, y, z, q0, qx, qy, qz).
    Returns:
        egomotions (np.ndarray): [n_frames x 7] each row is the camera egomotion (pose change) (x, y, z, q0, qx, qy, qz).
    """
    egomotions = infer_egomotions(torch.from_numpy(cam_poses))
    return egomotions.numpy()


# ----------------------------------------------------------------------


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


# def quaternions_to_rotation_matrices(quaternions: np.ndarray) -> np.ndarray:
#     """
#     Converts unit-quaternions (real-fist format) to a 3x3 rotation matrices.
#     Args:
#         quaternion: [N x 4] unit-quaternions (real-fist format) representing rotations.
#     Returns:
#         rot_mats [N x 3 x 3] rotation matrices.
#     """
#     assert quaternions.shape[1] == 4
#     quaternions.shape[0]
#     # convert to scipy format:  (w, x, y, z) -> (x, y, z, w)
#     quant_real_last = quaternions[:, [1, 2, 3, 0]]
#     rot_mats = spr.from_quat(quant_real_last).as_matrix()
#     return rot_mats
# # ----------------------------------------------------------------------

# def rotation_matrices_to_quaternions(rot_mats: np.ndarray) -> np.ndarray:
#     """"
#     Converts rotation matrices to unit-quaternions (real-fist format).
#     Args:
#         rot_mats [N x 3 x 3] rotation matrices.
#     Returns:
#         quaternion: [N x 4] unit-quaternions (real-fist format) representing rotations.
#     """
#     assert rot_mats.shape[1:] == (3, 3)
#     # convert to scipy format:  (w, x, y, z) -> (x, y, z, w)
#     quant_real_last = spr.from_matrix(rot_mats).as_quat()
#     # convert to pytorch format: (x, y, z, w) -> (w, x, y, z)
#     quaternions = quant_real_last[:, [3, 0, 1, 2]]
#     return quaternions


# # ----------------------------------------------------------------------


# def translations_and_quaternions_to_matrices(translations: np.ndarray, quaternions: np.ndarray) -> np.ndarray:
#     """
#     Converts pairs of 3D translation vectors and rotations unit-quaternions to the corresponding 4x4 transformation matrices.

#     Args:
#         translations: [N x 3] the translation vectors.
#         quaternions: [N x 4] unit-quaternions (real-fist format) representing rotations.

#     Returns:
#         transform_mats: [N x 4 x 4] the corresponding transformation matrices.
#     """

#     # Convert the quaternion to a rotation matrix.
#     rotation_matrix = quaternions_to_rotation_matrices(quaternions)

#     # Create the translation matrix.
#     transformation_matrix = np.eye(4)
#     transformation_matrix[:, :3, 3] = translations
#     transformation_matrix[:, :3, :3] = rotation_matrix

#     return transformation_matrix


# # ----------------------------------------------------------------------


# def invert_transformation_mat(transform_mat: np.ndarray) -> np.ndarray:
#     """"
#     Inverts a 4x4 transformation matrix.
#     """
#     rot_mat = transform_mat[:3, :3]
#     trans_vec = transform_mat[:3, 3]
#     inv_transform_mat = np.eye(4)
#     inv_transform_mat[:3, :3] = rot_mat.T
#     inv_transform_mat[:3, 3] = -rot_mat.T @ trans_vec
#     return inv_transform_mat


# # ---------------------------------------------------------------------------------------------------------------------
