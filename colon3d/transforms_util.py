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


@torch.jit.script  # disable this for debugging
def project_world_to_image_normalized_coord(
    cur_points_3d: torch.Tensor,
    cur_cam_poses: torch.Tensor,
) -> torch.Tensor:
    """Convert 3-D points to 2-D by projecting onto images.
    assumes camera parameters set a rectilinear image transform from 3d to 2d (i.e., fisheye undistorting was done)
    Args:
        cur_points_3d : [n_points x 3] (units: mm)
        cur_cam_poses : [n_points x 7] each row is (x, y, z, q0, qx, qy, qz) where (x, y, z) is the translation [mm] and (q0, qx, qy , qz) is the unit-quaternion of the rotation.
    Returns:
        points_2d : [n_points x 2]  (units: normalized image coordinates)
    """
    assert cur_points_3d.shape[1] == 3, f"Points are not in 3D, {cur_points_3d.shape}."
    assert cur_cam_poses.shape[1] == 7, f"Cam poses are not in 7D, {cur_cam_poses.shape}."
    # Rotate & translate to camera system
    eps = 1e-20
    cam_loc = cur_cam_poses[:, 0:3]  # [n_points x 3]
    cam_rot = cur_cam_poses[:, 3:7]  # [n_points x 4]
    inv_cam_rot = invert_rotation(cam_rot)  # [n_points x 4]
    points_cam_sys = rotate(cur_points_3d - cam_loc, inv_cam_rot)
    # Perspective transform to 2d image-plane
    # (this transforms to normalized image coordinates, i.e., fx=1, fy=1, cx=0, cy=0)
    z_cam_sys = points_cam_sys[:, 2]  # [n_points x 1]
    x_wrt_axis = points_cam_sys[:, 0] / (z_cam_sys + eps)  # [n_points x 1]
    y_wrt_axis = points_cam_sys[:, 1] / (z_cam_sys + eps)  # [n_points x 1]
    points_2d_nrm = torch.stack((x_wrt_axis, y_wrt_axis), dim=1)  # [n_points x 2]
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
    assert points_nrm.shape[1] == 2, f"Points are not in 2D, {points_nrm.shape}."

    # normalized coordinate corresponds to fx=1, fy=1, cx=0, cy=0, so we can just multiply by z_depth to get 3d point in the camera system
    z_cam_sys = z_depths
    x_cam_sys = points_nrm[:, 0] * z_depths
    y_cam_sys = points_nrm[:, 1] * z_depths
    points_3d_cam_sys = torch.stack((x_cam_sys, y_cam_sys, z_cam_sys), dim=1)
    return points_3d_cam_sys


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
    assert cam_poses.shape[1] == 7, f"Cam poses are not in 7D, {cam_poses.shape}."
    cam_loc = cam_poses[:, 0:3]  # [n_points x 3] (units: mm)
    cam_rot = cam_poses[:, 3:7]  # [n_points x 4]

    # cam_rot is the rotation from world to camera system, so we need to invert it to transform points from camera system to world system
    inv_cam_rot = invert_rotation(cam_rot)  # [n_points x 4]
    #  translate & rotate to world system
    points_3d_world_sys = cam_loc + rotate(points_3d_cam_sys, inv_cam_rot)
    return points_3d_world_sys


# --------------------------------------------------------------------------------------------------------------------


def transform_points_in_world_sys_to_cam_sys(
    points_3d_world_sys: torch.Tensor,
    cam_poses: torch.Tensor,
) -> torch.Tensor:
    """Transforms points in 3D world system to a 3D points in camera system coordinates.
    Args:
        points_3d_world_sys: [n_points x 3]  (units: mm) 3D points in world coordinates
        cam_poses: [n_points x 7]  [n_points x 7] each row is (x, y, z, q0, qx, qy, qz) where (x, y, z) is the translation [mm] and (q0, qx, qy, qz) is the unit-quaternion of the rotation.
    Returns:
        points_3d_cam_sys: [n_points x 3]  (units: mm) 3D points in camera system coordinates
    """
    assert cam_poses.shape[1] == 7, f"Cam poses are not in 7D, {cam_poses.shape}."
    cam_loc = cam_poses[:, 0:3]  # [n_points x 3] (units: mm)
    cam_rot = cam_poses[:, 3:7]  # [n_points x 4]
    # translate & rotate to camera system
    points_3d_cam_sys = rotate(points_3d_world_sys - cam_loc, cam_rot)
    return points_3d_cam_sys


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

    x_nrm = (pixels_x - cam_K[0, 2]) / cam_K[0, 0]  # u_nrm = (u - cx) / fx
    y_nrm = (pixels_y - cam_K[1, 2]) / cam_K[1, 1]  # v_nrm = (v - cy) / fy
    points_nrm = np.stack((x_nrm, y_nrm), axis=1)
    return points_nrm

# --------------------------------------------------------------------------------------------------------------------

def unproject_image_normalized_coord_to_world(points_nrm: torch.Tensor, z_depths: torch.Tensor, cam_poses: torch.Tensor) -> torch.Tensor:
    """Transforms a normalized image coordinate, camera pose and a given z depth to a 3D point in world system coordinates.
    Args:
        points_nrm: [n_points x 2] (units: normalized image coordinates)
        z_depths: [n_points] (units: mm)
        cam_poses: [n_points x 7]  [n_points x 7] each row is (x, y, z, q0, qx, qy, qz) where (x, y, z) is the translation [mm] and (q0, qx, qy, qz) is the unit-quaternion of the rotation.
    Returns:
        points_3d_world_sys: [n_points x 3]  (units: mm) 3D points in worlds system coordinates
    """
    points3d_cam_sys = unproject_image_normalized_coord_to_cam_sys(points_nrm=points_nrm, z_depths=z_depths)
    points3d_world_sys = transform_points_in_cam_sys_to_world_sys(points_3d_cam_sys=points3d_cam_sys, cam_poses=cam_poses)
    return points3d_world_sys

# --------------------------------------------------------------------------------------------------------------------


def apply_pose_change(
    start_pose: torch.Tensor,
    pose_change: torch.Tensor,
) -> torch.Tensor:
    """Applies a pose change to a given pose. (both are given in the same coordinate system)"""
    start_loc = start_pose[:, 0:3]  # [n_points x 3]
    start_rot = start_pose[:, 3:7]  # [n_points x 4]
    change_loc = pose_change[:, 0:3]  # [n_points x 3]
    rot_change = pose_change[:, 3:7]  # [n_points x 4]
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
    prev_poses = cam_poses[:-1, :]
    cur_poses = cam_poses[1:, :]

    poses_changes = find_pose_change(poses1=prev_poses, poses2=cur_poses)
    # set the first egomotion to be the identity rotation and zero translation:
    egomotions = torch.zeros_like(cam_poses)
    egomotions[0, :3] = 0
    egomotions[0, 3:] = get_identity_quaternion()
    # set the rest of the egomotions:
    egomotions[1:, :] = poses_changes
    return egomotions


# --------------------------------------------------------------------------------------------------------------------
