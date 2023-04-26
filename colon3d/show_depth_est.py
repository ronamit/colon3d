import argparse
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from colon3d.data_util import VideoLoader
from colon3d.depth_util import DepthEstimator

matplotlib.use("Agg")  # use a non-interactive backend

# ---------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--example_path",
        type=str,
        default="data/my_videos/Example_4",
        help="path to the video",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="data/my_videos/Example_4/temp_debug",
        help=" path to the save outputs",
    )
    parser.add_argument(
        "--n_frames_lim",
        type=int,
        default=11,
        help="upper limit on the number of frames used, if 0 then all frames are used",
    )

    args = parser.parse_args()

    save_path = Path(args.save_path)

    # Load video and undistort it
    video_loader = VideoLoader(
        example_path=args.example_path,
        n_frames_lim=args.n_frames_lim,
        undistort_FOV_deg=100,
        undistort_frame_width=500,
        undistort_frame_height=500,
    )
    depth_estimator = DepthEstimator(
        example_path=args.example_path,
        vslam_undistorter=video_loader.undistorter,
    )

    frame_idx = 0

    frame_width = video_loader.undistort_frame_width
    frame_height = video_loader.undistort_frame_height
    depth_map = torch.zeros((frame_height, frame_width))
    n_points = frame_height * frame_width
    y_vals, x_vals = torch.meshgrid(torch.arange(frame_height), torch.arange(frame_width), indexing="ij")
    queried_point_2d = torch.stack([x_vals.flatten(), y_vals.flatten()], dim=1)

    # get rgb frame
    rgb_frame = video_loader.get_undistorted_frame(frame_idx)

    # fet depth map
    frame_indexes = np.ones((n_points), dtype=int) * frame_idx
    depth_per_point = depth_estimator.get_depth_at_2d_points(
        frame_indexes=frame_indexes, queried_points_2d=queried_point_2d,
    )
    depth_map = depth_per_point.reshape((frame_height, frame_width))
    # Save depth map
    file_path = str(save_path / f"example_depth_frame_{frame_idx}.png")
    plt.imsave(file_path, depth_map, cmap="inferno")
    print("Depth image saved to: ", file_path)

    # Save rgb frame
    file_path = str(save_path / f"example_rgb_frame_{frame_idx}.png")
    cv2.imwrite(file_path, rgb_frame)
    print("Image saved to: ", file_path)


# ---------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
