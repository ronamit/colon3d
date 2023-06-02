import argparse
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np

from colon3d.data_util import FramesLoader
from colon3d.depth_util import DepthAndEgoMotionLoader
from colon3d.general_util import create_folder_if_not_exists, save_plot_and_close
from colon3d.rotations_util import get_identity_quaternion_np
from colon3d.slam_util import get_frame_point_cloud
from colon3d.visuals.create_3d_obj import plot_cam_and_point_cloud


# --------------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--example_path",
        type=str,
        default="data/sim_data/SimData2/Seq_00000",
        help="Path to the sequence folder",
    )
    parser.add_argument(
        "--frame_time",
        type=float,
        default=-1,
        help="The time of the frame to plot [sec], if -1 then the frame_index will be used",
    )
    parser.add_argument(
        "--frame_index",
        type=float,
        default=33,
        help="The index of the frame to plot, if frame_time is not -1 then  frame_time will be used instead",
    )
    parser.add_argument(
        "--depth_source",
        type=str,
        default="ground_truth",
        help="The source of the depth-map, can be 'ground_truth' or 'estimated'",
    )
    args = parser.parse_args()
    example_path = Path(args.example_path)

    frames_loader = FramesLoader(
        sequence_path=example_path,
    )
    depth_loader = DepthAndEgoMotionLoader(
        example_path=example_path,
        source=args.depth_source,
    )

    example_path = Path(args.example_path)
    plots_path = create_folder_if_not_exists(example_path / "plots")
    fps = frames_loader.fps
    frame_idx = args.frame_index if args.frame_time == -1 else int(args.frame_time * fps)
    frame_name = f"Frame{frame_idx:04d}"

    # save the RGB frame
    bgr_frame = frames_loader.get_frame_at_index(frame_idx, color_type="bgr", frame_type="full")
    file_path = str((plots_path / f"{frame_name}.png").resolve())
    cv2.imwrite(file_path, bgr_frame)
    print(f"Saved RGB frame to {file_path}")

    # get the depth map
    z_depth_frame = depth_loader.get_depth_map_at_frame(frame_idx)

    # if ground-truth camera pose is available, use it for the point cloud plot
    if args.depth_source == "ground_truth":
        with h5py.File(example_path / "gt_depth_and_egomotion.h5", "r") as h5f:
            gt_cam_poses = h5f["cam_poses"][:]
            cam_pose = gt_cam_poses[frame_idx]
    else:
        # otherwise, set the camera at the origin, and looking at the z-axis direction
        cam_pose = np.zeros(7)
        cam_pose[3:] = get_identity_quaternion_np()

    plt.figure()
    plt.imshow(z_depth_frame, aspect="equal")
    plt.xlabel("x [pixels]")
    plt.ylabel("y [pixels]")
    save_plot_and_close(plots_path / f"{frame_name}_depth_{args.depth_source}.png")

    depth_info = depth_loader.depth_info
    K_of_depth_map = depth_info["K_of_depth_map"]
    fx, fy = K_of_depth_map[0, 0], K_of_depth_map[1, 1]
    frame_height, frame_width = z_depth_frame.shape[:2]
    fov_deg = 2 * np.rad2deg(np.arctan(0.5 * min(frame_width / fx, frame_height / fy)))

    points3d = get_frame_point_cloud(z_depth_frame=z_depth_frame, K_of_depth_map=K_of_depth_map, cam_pose=cam_pose)
    plot_cam_and_point_cloud(
        points3d=points3d,
        cam_pose=cam_pose,
        cam_fov_deg=fov_deg,
        verbose=True,
        save_path=plots_path / f"{frame_name}_point_cloud_{args.depth_source}.html",
    )


# --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
