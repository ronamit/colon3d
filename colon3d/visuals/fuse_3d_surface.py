"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""
import argparse
import pickle
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as spat_rot

from colon3d.util.data_util import SceneLoader
from colon3d.util.general_util import ArgsHelpFormatter
from colon3d.util.torch_util import to_default_type
from tsdf_fusion import fusion

# --------------------------------------------------------------------------------------------------------------------


def cam_pose_as_hom_matrix(pos: np.ndarray, rot_quat: np.ndarray) -> np.ndarray:
    """
    Changes the format of the camera pose (translation & orientation) from quaternion to 4x4 rigid transformation matrix
    Args:
        pos: the position of the camera in the world coordinate system [3]
        rot_quat: the orientation of the camera in the world coordinate system [4]
                as quaternion in real-first format. Each quaternion will be normalized to unit norm.
    """
    # change the quaternion to real-last format
    rot_quat = rot_quat[[1, 2, 3, 0]]
    # get the rotation matrix from the quaternion
    rot = spat_rot.from_quat(rot_quat)
    rot_mat = rot.as_matrix()
    # create the matrix
    hom_mat = np.eye(4)
    hom_mat[:3, :3] = rot_mat
    hom_mat[:3, 3] = pos
    return hom_mat


# --------------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(formatter_class=ArgsHelpFormatter)
    parser.add_argument(
        "--example_path",
        type=str,
        default="data/sim_data/Scene_00009_short/Examples/0000",
        help="path to the video",
    )
    parser.add_argument("--start_frame", type=int, default=0, help="the index of the first frame to plot")
    parser.add_argument("--end_frame", type=int, default=50, help="the index of the last frame to plot")
    args = parser.parse_args()
    print(f"args={args}")
    start_frame = args.start_frame
    end_frame = args.end_frame
    frames_tp_plot = list(range(start_frame, end_frame))
    n_frames = len(frames_tp_plot)
    example_path = Path(args.example_path)

    scene_loader = SceneLoader(
        scene_path=example_path,
    )
    example_path = Path(args.example_path)

    with h5py.File(example_path / "gt_3d_data.h5", "r") as h5f:
        gt_depth_maps = to_default_type(h5f["z_depth_map"][:], num_type="float_m")
        gt_cam_poses = to_default_type(h5f["cam_poses"][:])
    with (example_path / "gt_depth_info.pkl").open("rb") as file:
        depth_info = pickle.load(file)

    # The matrix of the camera intrinsics (camera_sys_hom_cord = cam_K_mat @ pixel_hom_cord) [3x3]
    K_of_depth_map = depth_info["K_of_depth_map"]

    # the spatial size of each voxel [mm] (if the number of voxels is too large then increase this value)
    voxel_size_mm = 0.5

    # ======================================================================================================== #
    # (Optional) This is an example of how to compute the 3D bounds
    # in world coordinates of the convex hull of all camera view
    # frustums in the dataset
    # ======================================================================================================== #
    print("Estimating voxel volume bounds...")
    vol_bnds = np.zeros((3, 2))
    for frame_idx in frames_tp_plot:
        # Read depth image and camera pose
        depth_im = gt_depth_maps[frame_idx]

        # 4x4 rigid transformation matrix
        cam_pose = gt_cam_poses[frame_idx]
        cam_loc = cam_pose[:3]  # translation
        cam_rot = cam_pose[3:]  # quaternion orientation
        cam_pose_mat = cam_pose_as_hom_matrix(cam_loc, cam_rot)

        # Compute camera view frustum and extend convex hull
        view_frust_pts = fusion.get_view_frustum(depth_im, K_of_depth_map, cam_pose_mat)
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
    print("Volume bounds [mm]: \n", vol_bnds)
    # ======================================================================================================== #

    # ======================================================================================================== #
    # Integrate
    # ======================================================================================================== #
    # Initialize voxel volume
    print("Initializing voxel volume...")
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=voxel_size_mm)

    # Loop through RGB-D images and fuse them together
    t0_elapse = time.time()
    for i, frame_idx in enumerate(frames_tp_plot):
        print("Fusing frame %d/%d" % (i + 1, n_frames))

        # Read RGB-D image and camera pose
        color_image = scene_loader.get_frame_at_index(frame_idx, frame_type="full")
        depth_im = gt_depth_maps[frame_idx]

        # 4x4 rigid transformation matrix
        cam_pose = gt_cam_poses[frame_idx]
        cam_loc = cam_pose[:3]  # translation
        cam_rot = cam_pose[3:]  # quaternion orientation
        cam_pose_mat = cam_pose_as_hom_matrix(cam_loc, cam_rot)

        # Integrate observation into voxel volume (assume color aligned with depth)
        tsdf_vol.integrate(color_image, depth_im, K_of_depth_map, cam_pose_mat, obs_weight=1.0)

    print(f"Average number of frames processed per second: {n_frames / (time.time() - t0_elapse):.2f}")

    # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
    mesh_filepath = example_path / "mesh.ply"
    print(f"Saving mesh to {mesh_filepath}...")
    verts, faces, norms, colors = tsdf_vol.get_mesh()
    fusion.meshwrite(mesh_filepath, verts, faces, norms, colors)

    # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
    pointcloud_filepath = example_path / "point_cloud.ply"
    print(f"Saving point cloud to {pointcloud_filepath}...")
    point_cloud = tsdf_vol.get_point_cloud()
    fusion.pcwrite(pointcloud_filepath, point_cloud)


# --------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------------------------------------------------
