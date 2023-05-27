import argparse
import pickle
from pathlib import Path

import cv2
import h5py
import matplotlib
import numpy as np

matplotlib.use('agg')

import matplotlib.pyplot as plt

from colon3d.data_util import VideoLoader
from colon3d.depth_util import DepthEstimator
from colon3d.detections_util import DetectionsTracker
from colon3d.general_util import save_plot_and_close
from colon3d.slam_util import get_frame_point_cloud
from colon3d.visuals.create_3d_obj import plot_cam_and_point_cloud


# --------------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--example_path",
        type=str,
        default="data/sim_data/ExampleSimOut/Seq_00000",
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
        default=0,
        help="The index of the frame to plot, if frame_time is not -1 then  frame_time will be used instead",
    )
    args = parser.parse_args()
    example_path = Path(args.example_path)

    video_loader = VideoLoader(
        example_path=example_path,
    )
    DetectionsTracker(
        example_path=example_path,
        video_loader=video_loader,
    )
    DepthEstimator(
        example_path=example_path,
    )

    example_path = Path(args.example_path)
    fps = video_loader.fps
    frame_idx = args.frame_index if args.frame_time == -1 else int(args.frame_time * fps)
    frame_name = f"Frame{frame_idx:04d}"

    # save the RGB frame
    bgr_frame = video_loader.get_frame_at_index(frame_idx, color_type="bgr", frame_type="full")
    file_path = str((example_path / f"{frame_name}.png").resolve())
    cv2.imwrite(file_path, bgr_frame)
    print(f"Saved RGB frame to {file_path}")

    # save the ground truth depth frame, and its point cloud
    gt_depth_file_path = example_path / "gt_depth_and_cam_poses.h5"
    if not gt_depth_file_path.exists():
        print("No ground truth depth file found, skipping this part")
        return
    
    with h5py.File(gt_depth_file_path, "r") as h5f:
        gt_depth_maps = h5f["z_depth_map"][:]
        gt_cam_poses = h5f["cam_poses"][:]
    z_depth_frame = gt_depth_maps[frame_idx]
    cam_pose = gt_cam_poses[frame_idx]

    plt.figure()
    plt.imshow(z_depth_frame, aspect="equal")
    save_plot_and_close(example_path / f"{frame_name}_true_depth.png")

    with (example_path / "depth_info.pkl").open("rb") as file:
        depth_info = pickle.load(file)
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
        save_path=example_path / f"{frame_name}_true_point_cloud.html",
    )


# --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
