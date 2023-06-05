import numpy as np
import torch

from colon3d.rotations_util import (
    apply_rotation_change,
    find_rotation_change,
    get_identity_quaternion,
    invert_rotation,
    rotate,
)
from colon3d.torch_util import np_func

# --------------------------------------------------------------------------------------------------------------------

def assert_2d_tensor(t: torch.Tensor, dim2: int):
    assert t.ndim == 2, f"Tensor should be [n x {dim2}]."
    assert t.shape[1] == dim2, f"Tensor should be [n x {dim2}]."
# --------------------------------------------------------------------------------------------------------------------

def assert_1d_tensor(t: torch.Tensor):
    assert t.ndim == 1, "Tensor should be 1D."
# --------------------------------------------------------------------------------------------------------------------

def transform_pixel_image_coords_to_normalized(
    pixels_x: np.ndarray,
    pixels_y: np.ndarray,
    cam_K: np.ndarray,
) -> np.ndarray:
    """Transforms pixel coordinates to normalized image coordinates.
    Assumes the camera is rectilinear with a given K matrix.
    Args:
        pixels_x : [n_points] (units: pixels)
        pixels_y : [n_points] (units: pixels)
        cam_K: [3 x 3] camera intrinsics matrix (we assume it is of the form [fx, 0, cx; 0, fy, cy; 0, 0, 1])
    Returns:
        points_nrm: [n_points x 2] (units: normalized image coordinates)
    """
    assert_1d_tensor(pixels_x)
    assert_1d_tensor(pixels_y)
    x_nrm = (pixels_x - cam_K[0, 2]) / cam_K[0, 0]  # u_nrm = (u - cx) / fx
    y_nrm = (pixels_y - cam_K[1, 2]) / cam_K[1, 1]  # v_nrm = (v - cy) / fy
    points_nrm = np.stack((x_nrm, y_nrm), axis=1)
    return points_nrm


# --------------------------------------------------------------------------------------------------------------------



def transform_points_in_cam_sys_to_world_sys(
    points_3d_cam_sys: torch.Tensor,
    cam_poses: torch.Tensor,
) -> torch.Tensor:
    """Transforms points in 3D camera system to a 3D points in world coordinates.
    Args:
        points_3d_cam_sys: [n_points x 3]  (units: mm) 3D points in camera system coordinates
        cam_poses: [n_points x 7]  [n_points x 7] each row is (x, y, z, q0, qx, qy, qz) where (x, y, z) is the translation [mm] and (q0, qx, qy, qz) is the unit-quaternion of the rotation.
    Returns:
        points_3d_world_sys: [n_points x 3]  (units: mm) 3D points in world coordinates
    """
    assert_2d_tensor(points_3d_cam_sys, 3)
    assert_2d_tensor(cam_poses, 7)
    cam_loc = cam_poses[:, 0:3]  # [n_points x 3] (units: mm)
    cam_rot = cam_poses[:, 3:7]  # [n_points x 4]

    # cam_rot is the rotation from world to camera system, so we need to invert it to transform points from camera system to world system
    inv_cam_rot = invert_rotation(cam_rot)  # [n_points x 4]
    #  translate & rotate to world system
    points_3d_world = cam_loc + rotate(points_3d_cam_sys, inv_cam_rot)
    return points_3d_world


# --------------------------------------------------------------------------------------------------------------------


def transform_points_in_world_sys_to_cam_sys(
    points_3d_world: torch.Tensor,
    cam_poses: torch.Tensor,
) -> torch.Tensor:
    """Transforms points in 3D world system to a 3D points in camera system coordinates.
    Args:
        points_3d_world_sys: [n_points x 3]  (units: mm) 3D points in world coordinates
        cam_poses: [n_points x 7]  [n_points x 7] each row is (x, y, z, q0, qx, qy, qz) where (x, y, z) is the translation [mm] and (q0, qx, qy, qz) is the unit-quaternion of the rotation.
    Returns:
        points_3d_cam_sys: [n_points x 3]  (units: mm) 3D points in camera system coordinates
    """
    assert_2d_tensor(points_3d_world, 3)
    assert_2d_tensor(cam_poses, 7)
    cam_loc = cam_poses[:, 0:3]  # [n_points x 3] (units: mm)
    cam_rot = cam_poses[:, 3:7]  # [n_points x 4]
    # translate & rotate to camera system
    points_3d_cam_sys = rotate(points_3d_world - cam_loc, cam_rot)
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


def apply_pose_change(
    start_pose: torch.Tensor,
    pose_change: torch.Tensor,
) -> torch.Tensor:
    """Applies a pose change to a given pose. (both are given in the same coordinate system)
    Args:
        start_pose: [n x 7] each row is (x, y, z, q0, qx, qy, qz) where (x, y, z) is the translation [mm] and (q0, qx, qy , qz) is the unit-quaternion of the rotation.
        pose_change: [n x 7] each row is (x, y, z, q0, qx, qy, qz) where (x, y, z) is the translation [mm] and (q0, qx, qy , qz) is the unit-quaternion of the rotation.
    Returns:
        final_pose: [n x 7] each row is (x, y, z, q0, qx, qy, qz) where (x, y, z) is the translation [mm] and (q0, qx, qy , qz) is the unit-quaternion of the rotation.
    """
    assert start_pose.ndim == 2, f"Start pose is not 2D, {start_pose.ndim}."
    assert start_pose.shape[1] == 7, f"Start pose is not in 7D, {start_pose.shape}."
    start_loc = start_pose[:, 0:3]  # [n x 3]
    start_rot = start_pose[:, 3:7]  # [n x 4]
    change_loc = pose_change[:, 0:3]  # [n x 3]
    rot_change = pose_change[:, 3:7]  # [n x 4]
    final_loc = start_loc + change_loc
    final_rot = apply_rotation_change(start_rot=start_rot, rot_change=rot_change)
    final_pose = torch.cat((final_loc, final_rot), dim=1)
    return final_pose


# --------------------------------------------------------------------------------------------------------------------


def find_pose_change(
    start_pose: torch.Tensor,
    final_pose: torch.Tensor,
) -> torch.Tensor:
    """Finds the pose change that transforms start_pose to final_pose. (both are given in the same coordinate system).
    Args:
        start_pose: [n_points x 7] each row is (x, y, z, q0, qx, qy, qz) where (x, y, z) is the translation [mm] and (q0, qx, qy, qz) is the unit-quaternion of the rotation.
        final_pose: [n_points x 7] each row is (x, y, z, q0, qx, qy, qz) where (x, y, z) is the translation [mm] and (q0, qx, qy, qz) is the unit-quaternion of the rotation.
    Returns:
        pose_change: [n_points x 7] each row is (x, y, z, q0, qx, qy, qz) where (x, y, z) is the translation [mm] and (q0, qx, qy, qz) is the unit-quaternion of the rotation.
    """
    assert_2d_tensor(start_pose, 7)
    assert_2d_tensor(final_pose, 7)
    start_loc = start_pose[:, 0:3]  # [n_points x 3]
    start_rot = start_pose[:, 3:7]  # [n_points x 4]
    final_loc = final_pose[:, 0:3]  # [n_points x 3]
    final_rot = final_pose[:, 3:7]  # [n_points x 4]
    change_loc = final_loc - start_loc
    rot_change = find_rotation_change(start_rot=start_rot, final_rot=final_rot)
    pose_change = torch.cat((change_loc, rot_change), dim=1)
    return pose_change


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
    assert_2d_tensor(cam_pose, 7)
    frame_width, frame_height = z_depth_frame.shape
    n_pix = frame_width * frame_height
    # find the world coordinates that each pixel in the depth map corresponds to:
    pixels_y, pixels_x = np.meshgrid(np.arange(frame_height), np.arange(frame_width))
    # notice that the depth image coordinates are (y,x) not (x,y).
    z_depths = z_depth_frame.flatten()
    pixels_x = pixels_x.flatten()
    pixels_y = pixels_y.flatten()

    points_nrm = transform_pixel_image_coords_to_normalized(
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

    poses_changes = find_pose_change(start_pose=prev_poses, final_pose=cur_poses)
    # set the first egomotion to be the identity rotation and zero translation:
    egomotions = torch.zeros_like(cam_poses)
    egomotions[0, :3] = 0
    egomotions[0, 3:] = get_identity_quaternion()
    # set the rest of the egomotions:
    egomotions[1:, :] = poses_changes
    return egomotions


# --------------------------------------------------------------------------------------------------------------------
