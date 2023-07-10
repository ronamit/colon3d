import numpy as np
import torch

from colon3d.utils.camera_util import CamInfo
from colon3d.utils.rotations_util import (
    compose_rotations,
    get_identity_quaternion,
    invert_rotation,
    normalize_quaternions,
    rotate_points,
)
from colon3d.utils.torch_util import (
    assert_1d_tensor,
    assert_2d_tensor,
    get_default_dtype,
    get_device,
    np_func,
    to_numpy,
)

# --------------------------------------------------------------------------------------------------------------------


def get_identity_transform():
    """Returns the identity pose transform (no change)"""
    return torch.cat((torch.zeros(3), get_identity_quaternion()), dim=0).to(get_default_dtype()).to(get_device())


# --------------------------------------------------------------------------------------------------------------------


def transform_rectilinear_image_pixel_coords_to_normalized(
    pixels_x: np.ndarray,
    pixels_y: np.ndarray,
    cam_K: np.ndarray,
) -> np.ndarray:
    """Transforms pixel coordinates to normalized image coordinates.
    Assumes the camera is rectilinear with a given K matrix (no fisheye distortion)
    Args:
        pixels_x : [n_points] (units: pixels)
        pixels_y : [n_points] (units: pixels)
        cam_K: [3 x 3] camera intrinsics matrix (we assume it is of the form [fx, 0, cx; 0, fy, cy; 0, 0, 1])
    Returns:
        points_nrm: [n_points x 2] (units: normalized image coordinates)
    """
    assert_1d_tensor(pixels_x)
    assert_1d_tensor(pixels_y)
    dtype = get_default_dtype("numpy")
    x_nrm = (pixels_x - cam_K[0, 2]) / cam_K[0, 0]  # u_nrm = (u - cx) / fx
    y_nrm = (pixels_y - cam_K[1, 2]) / cam_K[1, 1]  # v_nrm = (v - cy) / fy
    points_nrm = np.stack((x_nrm, y_nrm), axis=1, dtype=dtype)
    return points_nrm


# --------------------------------------------------------------------------------------------------------------------


def transform_rectilinear_image_norm_coords_to_pixel(
    points_nrm: np.ndarray,
    cam_info: CamInfo | None = None,
    cam_K: np.ndarray | None = None,
    im_height: int | None = None,
    im_width: int | None = None,
) -> np.ndarray:
    """Transforms normalized image coordinates to pixel coordinates.
    Assumes the camera is rectilinear with a given K matrix (no fisheye distortion)
    Args:
        points_nrm: [n_points x 2] (units: normalized image coordinates)
        cam_K: [3 x 3] camera intrinsics matrix (we assume it is of the form [fx, 0, cx; 0, fy, cy; 0, 0, 1])
    Returns:
        points_pix: [n_points x 2] (units: pixels)
    """
    points_nrm = to_numpy(points_nrm)
    if cam_info is not None:
        fx = cam_info.fx
        fy = cam_info.fy
        cx = cam_info.cx
        cy = cam_info.cy

    elif cam_K is not None:
        fx = cam_K[0, 0]
        fy = cam_K[1, 1]
        cx = cam_K[0, 2]
        cy = cam_K[1, 2]
    x_pix = points_nrm[:, 0] * fx + cx  # u = u_nrm * fx + cx
    y_pix = points_nrm[:, 1] * fy + cy  # v = v_nrm * fy + cy
    points_pix = np.stack((x_pix, y_pix), axis=1)

    # Round to nearest pixel
    points_pix = points_pix.round().astype(int)

    if im_height is not None:
        # clip the y value
        points_pix[:, 1] = np.clip(points_pix[:, 1], 0, im_height - 1)

    if im_width is not None:
        # clip the x value
        points_pix[:, 0] = np.clip(points_pix[:, 0], 0, im_width - 1)

    return points_pix


# --------------------------------------------------------------------------------------------------------------------


def transform_points_in_cam_sys_to_world_sys(
    points_3d_cam_sys: torch.Tensor,
    cam_poses: torch.Tensor,
) -> torch.Tensor:
    """Transforms points in 3D camera system to a 3D points in world coordinates.
        Note that cam_poses is the the transformation from world to camera system.
    Args:
        points_3d_cam_sys: [n_points x 3]  (units: mm) 3D points in camera system coordinates
        cam_poses: [n_points x 7]  [n_points x 7] each row is (x, y, z, q0, qx, qy, qz) where (x, y, z) is the translation [mm] and (q0, qx, qy, qz) is the unit-quaternion of the rotation.
    Returns:
        points_3d_world_sys: [n_points x 3]  (units: mm) 3D points in world coordinates
    """
    assert_2d_tensor(points_3d_cam_sys, 3)
    cam_trans = cam_poses[:, 0:3]  # [n_points x 3] (units: mm)
    cam_rot = cam_poses[:, 3:7]  # [n_points x 4]  (unit-quaternion)
    # get the inverse of the camera rotation : R^{-1}
    inv_cam_rot = invert_rotation(cam_rot)
    # apply the transform to the points : (R^{-1} @ point3d) + t
    points_3d_world = cam_trans + rotate_points(points_3d_cam_sys, inv_cam_rot)
    return points_3d_world


# --------------------------------------------------------------------------------------------------------------------


def transform_points_in_world_sys_to_cam_sys(
    points_3d_world: torch.Tensor,
    cam_poses: torch.Tensor,
) -> torch.Tensor:
    """Transforms points in 3D world system to a 3D points in camera system coordinates.
        Note that cam_poses is the the transformation from world to camera system.
    Args:
        points_3d_world_sys: [n_points x 3]  (units: mm) 3D points in world coordinates
        cam_poses: [n_points x 7]  [n_points x 7] each row is (x, y, z, q0, qx, qy, qz) where (x, y, z) is the translation [mm] and (q0, qx, qy, qz) is the unit-quaternion of the rotation.
    Returns:
        points_3d_cam_sys: [n_points x 3]  (units: mm) 3D points in camera system coordinates
    """
    if points_3d_world.ndim == 1:
        points_3d_world = points_3d_world.unsqueeze(0)
    if cam_poses.ndim == 1:
        cam_poses = cam_poses.unsqueeze(0)
    assert_2d_tensor(cam_poses, 7)
    cam_trans = cam_poses[:, 0:3]  # [n_points x 3] (units: mm)
    cam_rot = cam_poses[:, 3:7]  # [n_points x 4]  (unit-quaternion)

 
    # translate & rotate to camera system  R @ (point3d - t)
    points_3d_cam_sys = rotate_points(points_3d_world - cam_trans, cam_rot)
    return points_3d_cam_sys


# --------------------------------------------------------------------------------------------------------------------


def unproject_image_normalized_coord_to_world(
    points_nrm: torch.Tensor,
    z_depths: torch.Tensor,
    cam_poses: torch.Tensor,
) -> torch.Tensor:
    """Transforms a normalized image coordinate, camera pose and a given z depth to a 3D point in world system coordinates.
    Args:
        points_nrm: [n_points x 2] (units: normalized image coordinates)
        z_depths: [n_points] (units: mm)
        cam_poses: [n_points x 7]  [n_points x 7] each row is (x, y, z, q0, qx, qy, qz) where (x, y, z) is the translation [mm] and (q0, qx, qy, qz) is the unit-quaternion of the rotation.
    Returns:
        points_3d_world_sys: [n_points x 3]  (units: mm) 3D points in worlds system coordinates
    """
    assert_2d_tensor(points_nrm, 2)
    assert_1d_tensor(z_depths)
    assert_2d_tensor(cam_poses, 7)
    points3d_cam_sys = unproject_image_normalized_coord_to_cam_sys(points_nrm=points_nrm, z_depths=z_depths)
    points3d_world_sys = transform_points_in_cam_sys_to_world_sys(
        points_3d_cam_sys=points3d_cam_sys,
        cam_poses=cam_poses,
    )
    return points3d_world_sys


# --------------------------------------------------------------------------------------------------------------------


def project_cam_sys_to_image_normalized_coord(
    points3d_cam_sys: torch.Tensor,
) -> torch.Tensor:
    """Projects 3D points in camera system coordinates to normalized image coordinates.
    Args:
        points_3d_cam_sys: [n_points x 3]  (units: mm) 3D points in camera system coordinates
    Returns:
        points_2d_nrm: [n_points x 2] (units: normalized image coordinates)
    Notes:
        No camera intrinsics are needed since we transform to normalized image coordinates (K = I)
    """
    assert_2d_tensor(points3d_cam_sys, 3)
    # Perspective transform to 2d image-plane
    # (this transforms to normalized image coordinates, i.e., fx=1, fy=1, cx=0, cy=0)
    z_cam_sys = points3d_cam_sys[:, 2]  # [n_points x 1]
    x_nrm = points3d_cam_sys[:, 0] / z_cam_sys  # [n_points x 1]
    y_nrm = points3d_cam_sys[:, 1] / z_cam_sys  # [n_points x 1]
    points_2d_nrm = torch.stack((x_nrm, y_nrm), dim=1)  # [n_points x 2]
    return points_2d_nrm


# --------------------------------------------------------------------------------------------------------------------


# @torch.jit.script  # disable this for debugging
def project_world_to_image_normalized_coord(
    points3d_world: torch.Tensor,
    cam_poses: torch.Tensor,
) -> torch.Tensor:
    """Projects 3D points in world system coordinates to normalized image coordinates.
    Args:
        points3d_world : [n_points x 3] (units: mm) 3D points in world coordinates
        cam_poses : [n_points x 7] each row is (x, y, z, q0, qx, qy, qz) where (x, y, z) is the translation [mm] and (q0, qx, qy , qz) is the unit-quaternion of the rotation.
    Returns:
        points_2d : [n_points x 2]  (units: normalized image coordinates)
    Notes:
        Assumes camera parameters set a rectilinear image transform from 3d to 2d (i.e., fisheye undistorting was done)
    """
    assert_2d_tensor(points3d_world, 3)
    assert_2d_tensor(cam_poses, 7)

    # Translate & rotate to camera system
    points3d_cam_sys = transform_points_in_world_sys_to_cam_sys(
        points_3d_world=points3d_world,
        cam_poses=cam_poses,
    )
    # Perspective transform to 2d image-plane
    points_2d_nrm = project_cam_sys_to_image_normalized_coord(points3d_cam_sys=points3d_cam_sys)
    return points_2d_nrm


# --------------------------------------------------------------------------------------------------------------------


def unproject_image_normalized_coord_to_cam_sys(
    points_nrm: torch.Tensor,
    z_depths: torch.Tensor,
) -> torch.Tensor:
    """Transforms a normalized image coordinate and a given z depth to a 3D point in camera system coordinates.
    Args:
        points_nrm: [n_points x 2] (units: normalized image coordinates)
        z_depths: [n_points] (units: mm)
    Returns:
        points_3d_cam_sys: [n_points x 3]  (units: mm) 3D points in camera system coordinates
    """
    assert_2d_tensor(points_nrm, 2)
    assert_1d_tensor(z_depths)

    # normalized coordinate corresponds to fx=1, fy=1, cx=0, cy=0, so we can just multiply by z_depth to get 3d point in the camera system
    z_cam_sys = z_depths
    x_cam_sys = points_nrm[:, 0] * z_depths
    y_cam_sys = points_nrm[:, 1] * z_depths
    points_3d_cam_sys = torch.stack((x_cam_sys, y_cam_sys, z_cam_sys), dim=1)
    return points_3d_cam_sys


# --------------------------------------------------------------------------------------------------------------------


def compose_poses(
    pose1: torch.Tensor,
    pose2: torch.Tensor,
) -> torch.Tensor:
    """Composes two pose transform (pose1 applied first, then pose2) to a single equivalent pose transform (both are given in the same coordinate system)
        We assume the transformation is applied in the following order:
        If Pose1 = [R1 | t1] and PoseChange = [R2 | t2] [in 4x4 matrix format],
        then the final pose is Pose2 = [R2 @ R1 | R2 @ t1 + t2]
    Args:
        pose1: [n x 7] each row is (x, y, z, q0, qx, qy, qz) where (x, y, z) is the translation [mm] and (q0, qx, qy , qz) is the unit-quaternion of the rotation.
        pose2: [n x 7] each row is (x, y, z, q0, qx, qy, qz) where (x, y, z) is the translation [mm] and (q0, qx, qy , qz) is the unit-quaternion of the rotation.
    Returns:
        pose_tot: [n x 7] each row is (x, y, z, q0, qx, qy, qz) where (x, y, z) is the translation [mm] and (q0, qx, qy , qz) is the unit-quaternion of the rotation.
    """
    if pose1.ndim == 1:
        pose1 = pose1.unsqueeze(0)
    if pose2.ndim == 1:
        pose2 = pose2.unsqueeze(0)
    assert pose1.ndim == 2, f"Start pose is not 2D, {pose1.ndim}."
    assert pose1.shape[1] == 7, f"Start pose is not in 7D, {pose1.shape}."
    trans1 = pose1[:, 0:3]  # [n x 3]
    rot1 = pose1[:, 3:7]  # [n x 4]
    trans2 = pose2[:, 0:3]  # [n x 3]
    rot2 = pose2[:, 3:7]  # [n x 4]
    #  get R2 @ t1
    rot2_trans1 = rotate_points(points3d=trans1, rot_vecs=rot2)  # [n x 3]
    # get R2 @ t1 + t2
    trans_tot = rot2_trans1 + trans2
    # get R2 @ R1
    rot_tot = compose_rotations(rot1=rot1, rot2=rot2)  #   [n x 4]
    pose_tot = torch.cat((trans_tot, rot_tot), dim=1)
    return pose_tot


# --------------------------------------------------------------------------------------------------------------------


def get_inverse_pose(
    pose: torch.Tensor,
) -> torch.Tensor:
    """Returns the inverse pose of a given pose. (both are given in the same coordinate system)
        If Pose = [R | t] [in 4x4 matrix format], then the inverse pose  [R^{-1} | -R^{-1} @ t]
    Args:
        pose: [n x 7] each row is (x, y, z, q0, qx, qy, qz) where (x, y, z) is the translation [mm] and (q0, qx, qy , qz) is the unit-quaternion of the rotation.
    Returns:
        inv_pose: [n x 7] each row is (x, y, z, q0, qx, qy, qz) where (x, y, z) is the translation [mm] and (q0, qx, qy , qz) is the unit-quaternion of the rotation.
    """
    assert pose.ndim == 2, f"Pose is not 2D, {pose.ndim}."
    trans = pose[:, 0:3]  # [n x 3]
    rot = pose[:, 3:7]  # [n x 4]
    # get R^{-1}
    inv_rot = invert_rotation(rot)  # [n x 4]
    # get -R^{-1} @ t
    inv_loc = rotate_points(points3d=-trans, rot_vecs=inv_rot)  # [n x 3]
    inv_pose = torch.cat((inv_loc, inv_rot), dim=1)
    return inv_pose


# --------------------------------------------------------------------------------------------------------------------


def get_pose_delta(
    pose1: torch.Tensor,
    pose2: torch.Tensor,
) -> torch.Tensor:
    """Finds the pose that when applied after pose1 gives in total the transform of pose2. (both are given in the same coordinate system).
        If  Pose1 = [R1 | t1] and Pose2 = [R2 | t2]  [in 4x4 matrix format],
        then the pose_delta is given by:
        PoseDelta = Pose2 @ (Pose1)^(-1) = [R2 @ (R1)^(-1) | t2 - R2 @ (R1)^(-1) @ t1]
    Args:
        pose1: [n_points x 7] each row is (x, y, z, q0, qx, qy, qz) where (x, y, z) is the translation [mm] and (q0, qx, qy, qz) is the unit-quaternion of the rotation.
        pose2: [n_points x 7] each row is (x, y, z, q0, qx, qy, qz) where (x, y, z) is the translation [mm] and (q0, qx, qy, qz) is the unit-quaternion of the rotation.
    Returns:
        pose_delta: [n_points x 7] each row is (x, y, z, q0, qx, qy, qz) where (x, y, z) is the translation [mm] and (q0, qx, qy, qz) is the unit-quaternion of the rotation.
    """
    if pose1.ndim == 1:
        pose1 = pose1.unsqueeze(dim=0)
    if pose2.ndim == 1:
        pose2 = pose2.unsqueeze(dim=0)
    assert_2d_tensor(pose1, 7)
    assert_2d_tensor(pose2, 7)
    assert pose1.shape[0] == pose2.shape[0]
    # get (Pose1)^(-1) = [R1^{-1} | -R1^{-1} @ t1]
    inv_pose1 = get_inverse_pose(pose=pose1)  # [n_points x 7]
    # get Pose2 @ (Pose1)^(-1) = [R2 @ (R1)^(-1) | t2 - R2 @ (R1)^(-1) @ t1]
    pose_delta = compose_poses(pose1=inv_pose1, pose2=pose2)  # [n_points x 7]
    
    return pose_delta


# --------------------------------------------------------------------------------------------------------------------
def get_frame_point_cloud(z_depth_frame: np.ndarray, K_of_depth_map: np.ndarray, cam_pose: np.ndarray):
    """Returns a point cloud for a given depth map and camera pose
    Args:
        z_depth_frame: [frame_width x frame_height] (units: mm)
        K_of_depth_map: [3 x 3] camera intrinsics matrix (we assume it is of the form [fx, 0, cx; 0, fy, cy; 0, 0, 1])
        cam_pose: [7] each row is (x, y, z, q0, qx, qy, qz) where (x, y, z) is the translation [mm] and (q0, qx, qy, qz) is the unit-quaternion of the rotation (in the world coordinate system)
            If None, then we assume the camera is at the origin and looking in the direction of the positive z axis.
    Returns:
        points3d: [n_points x 3]  (units: mm)

    """
    if cam_pose is None:
        cam_pose = np_func(get_identity_transform)()
    if cam_pose.ndim == 1:
        cam_pose = cam_pose[np.newaxis, :]
    assert_2d_tensor(cam_pose, 7)
    frame_width, frame_height = z_depth_frame.shape
    n_pix = frame_width * frame_height
    # find the world coordinates that each pixel in the depth map corresponds to:
    pixels_y, pixels_x = np.meshgrid(np.arange(frame_height), np.arange(frame_width), indexing="ij")
    # notice that the depth image coordinates are (y,x) not (x,y).
    z_depths = z_depth_frame.flatten()
    pixels_x = pixels_x.flatten()
    pixels_y = pixels_y.flatten()

    points_nrm = transform_rectilinear_image_pixel_coords_to_normalized(
        pixels_x=pixels_x,
        pixels_y=pixels_y,
        cam_K=K_of_depth_map,
    )
    points3d_cam_sys = np_func(unproject_image_normalized_coord_to_cam_sys)(points_nrm=points_nrm, z_depths=z_depths)

    if cam_pose is None:
        return points3d_cam_sys

    points3d_world_sys = np_func(transform_points_in_cam_sys_to_world_sys)(
        points_3d_cam_sys=points3d_cam_sys,
        cam_poses=np.tile(cam_pose, (n_pix, 1)),
    )
    return points3d_world_sys


# --------------------------------------------------------------------------------------------------------------------


def infer_egomotions(cam_poses: torch.Tensor):
    """infer the egomotions from the previous and current camera poses
    Args:
        cam_poses (torch.Tensor): [n_frames x 7] each row is the camera pose (x, y, z, q0, qx, qy, qz).
    Returns:
        egomotions (torch.Tensor): [n_frames x 7] each row is the camera egomotion (pose change) (x, y, z, q0, qx, qy, qz).
    """
    assert_2d_tensor(cam_poses, 7)
    prev_poses = cam_poses[:-1, :]
    cur_poses = cam_poses[1:, :]

    poses_changes = get_pose_delta(pose1=prev_poses, pose2=cur_poses)

    # set the first egomotion to be the identity rotation and zero translation:
    egomotions = torch.zeros_like(cam_poses)
    egomotions[0, :3] = 0
    egomotions[0, 3:] = get_identity_quaternion()
    # set the rest of the egomotions:
    egomotions[1:, :] = poses_changes
    return egomotions


# --------------------------------------------------------------------------------------------------------------------


def find_rigid_registration(poses1: np.ndarray, poses2: np.ndarray, method: str = "first_frame"):
    """Finds the rigid registration that aligns poses1 to poses2.
        The rigid registration is a 6-DoF transformation (3 for translation and 3 for rotation), which is represented as a 7-vector (x, y, z, q0, qx, qy, qz).
        If P1 and P2 are the poses in 4x4 homogeneous coordinates, then the rigid registration is the transformation that aims to approximate P2 =  P1 @ rigid_registration.
    Args:
        poses1: (N, 7) array of poses.
        poses2: (N, 7) array of poses.
    Returns:
        rigid_align: the rigid registration that aligns poses1 to poses2.
    Notes:
    """

    if method == "first_frame":
        #  Find the pose change in the first frame of poses1 to the first frame of poses2.
        rigid_align = get_pose_delta(pose1=poses1[0], pose2=poses2[0])

    # elif method == "Kabsch":
    #     # * Uses the Kabsch-Umeyama algorithm to find the rigid registration.
    #     # minimize ||points2 - (points1 * R + t)||^2 - where R is a rotation matrix and t is a translation vector.
    #     # does not take into account the rotations of the poses, only the locations
    #     # See:
    #     # https://en.wikipedia.org/wiki/Kabsch_algorithm
    #     # https://nghiaho.com/?page_id=671
    #     # https://github.com/nghiaho12/rigid_transform_3D

    #     points1 = poses1[:, :3]  # (N, 3)
    #     points2 = poses2[:, :3]  # (N, 3)

    #     # Compute the centroids of each point set
    #     centroid1 = np.mean(points1, axis=0)
    #     centroid2 = np.mean(points2, axis=0)

    #     #  # Center the point sets by subtracting the centroids
    #     centered1 = points1 - centroid1
    #     centered2 = points2 - centroid2

    #     # Compute the covariance matrix
    #     H = centered1.T @ centered2

    #     # Perform singular value decomposition (SVD) on the covariance matrix
    #     U, _, Vt = np.linalg.svd(H)

    #     # Compute the rotation matrix using the SVD results
    #     R = Vt.T @ U.T

    #     # Handle the special case of reflections
    #     if np.linalg.det(R) < 0:
    #         Vt[-1, :] *= -1
    #         R = Vt.T @ U.T

    #     # Compute the translation vector
    #     align_trans = centroid2 - R @ centroid1

    #     # transform the rotation matrix to a unit quaternion
    #     align_rot = scipy.spatial.transform.Rotation.from_matrix(R).as_quat()
    #     # change to real-first form quaternion (qw, qx, qy, qz)
    #     align_rot = align_rot[[3, 0, 1, 2]]
    #     align_rot = np_func(normalize_quaternions)(align_rot)
    #     rigid_align = torch.cat((align_trans, align_rot))

    else:
        raise ValueError(f"Unknown method: {method}")
    return rigid_align


# --------------------------------------------------------------------------------------------------------------------



def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    """
The transformation matrices returned from the functions in this file assume
the points on which the transformation will be applied are column vectors.
i.e. the R matrix is structured as

    R = [
            [Rxx, Rxy, Rxz],
            [Ryx, Ryy, Ryz],
            [Rzx, Rzy, Rzz],
        ]  # (3, 3)

This matrix can be applied to column vectors by post multiplication
by the points e.g.

    points = [[0], [1], [2]]  # (3 x 1) xyz coordinates of a point
    transformed_points = R * points

To apply the same matrix to points which are row vectors, the R matrix
can be transposed and pre multiplied by the points:

e.g.
    points = [[0, 1, 2]]  # (1 x 3) xyz coordinates of a point
    transformed_points = points * R.transpose(1, 0)
"""
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))
# --------------------------------------------------------------------------------------------------------------------


def main():
    # set random seed:
    torch.manual_seed(0)

    # create random camera poses:
    n = 4
    poses = torch.rand(n, 7, dtype=torch.float64)

    print("original poses:")
    for i in range(n):
        poses[i, 3:] = normalize_quaternions(poses[i, 3:])
        print("i=", i, "loc=", poses[i, :3], "rot=", poses[i, 3:])

    #  infer_egomotions:
    egomotions = infer_egomotions(cam_poses=poses)
    print("egomotions:")
    for i in range(n):
        print("i=", i, "egomotion loc.=", egomotions[i, :3], " rot.=", egomotions[i, 3:])

    # Reconstruct the camera poses from the egomotions:
    poses_rec = torch.zeros_like(poses)
    poses_rec[0, 3:] = get_identity_quaternion()
    for i in range(1, n):
        poses_rec[i, :] = compose_poses(
            pose1=poses_rec[i - 1, :].unsqueeze(0),
            pose2=egomotions[i, :].unsqueeze(0),
        )
    print("reconstructed poses:")
    for i in range(n):
        print("i=", i, "rec. loc=", poses_rec[i, :3], "rot=", poses_rec[i, 3:])

    # Find rigid transform that aligns the reconstructed poses with the original poses:
    rigid_align = find_rigid_registration(poses1=poses_rec, poses2=poses)
    print("rigid_align_loc=", rigid_align[0][:3], "rigid_align_rot=", rigid_align[0][3:])

    print("aligned reconstructed poses:")
    poses_rec_aligned = torch.zeros_like(poses)
    # apply the alignment to the reconstructed poses:
    for i in range(n):
        poses_rec_aligned[i] = compose_poses(pose1=rigid_align, pose2=poses_rec[i])

    # print the results:
    for i in range(n):
        print("i=", i, "rec. align. loc=", poses_rec_aligned[i, :3], "rot=", poses_rec_aligned[i, 3:])


# --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------------------------------------------------
