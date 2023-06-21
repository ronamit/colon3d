import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from colon3d.data_util import SceneLoader
from colon3d.depth_egomotion import DepthAndEgoMotionLoader
from colon3d.general_util import ArgsHelpFormatter, Tee, create_empty_folder, save_plot_and_close
from colon3d.torch_util import to_numpy

# ---------------------------------------------------------------------------------------------------------------------
# plot for each example - the first frame ground truth and estimated of depth maps
# and calculate the mean depth the value of the estimated depth map and of the ground truth depth map
# ---------------------------------------------------------------------------------------------------------------------


def examine_depths(depth_loader: DepthAndEgoMotionLoader, frames_loader: SceneLoader, save_path: Path, scene_name: str, depth_source: str):
    # save a heatmap of the first frame depth map
    # and RGB image of the first frame
    frame_idx = 0
    rgb_frame = frames_loader.get_frame_at_index(frame_idx=frame_idx)
    depth_map = depth_loader.get_depth_map_at_frame(frame_idx=frame_idx, rgb_frame=rgb_frame)
    depth_map = to_numpy(depth_map)
    plt.figure()
    plt.imshow(depth_map)
    plt.colorbar()
    fig_name = f"{scene_name}_depth_{depth_source}_{frame_idx}"
    plt.title(fig_name)
    plt.tight_layout()
    save_plot_and_close(save_path / fig_name)

    plt.figure()
    plt.imshow(rgb_frame)
    fig_name = f"{scene_name}_rgb_{frame_idx}"
    plt.title(fig_name)
    save_plot_and_close(save_path / fig_name)

    # calculate the mean depth of all frames
    n_frames = frames_loader.n_frames
    all_frames_avg_depth = np.zeros(n_frames)
    for i in range(n_frames):
        rgb_frame = frames_loader.get_frame_at_index(frame_idx=i)
        depth_map = depth_loader.get_depth_map_at_frame(frame_idx=i, rgb_frame=rgb_frame)
        depth_map = to_numpy(depth_map)
        all_frames_avg_depth[i] = np.mean(depth_map)
    scene_avg_depth = np.mean(all_frames_avg_depth)
    return scene_avg_depth


# ---------------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(formatter_class=ArgsHelpFormatter)
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/sim_data/SimData9",
        help="Path to the dataset of scenes.",
    )
    args = parser.parse_args()
    dataset_path = Path(args.dataset_path).expanduser()
    with Tee(dataset_path / "examine_depths.log"):
        scenes_paths = list(dataset_path.glob("Scene_*"))
        n_scenes = len(scenes_paths)
        print(f"n_scenes = {n_scenes}")
        scenes_paths.sort()
        save_path = dataset_path / "DepthsExamination"
        create_empty_folder(save_path)
        scene_avg_gt_depth = np.zeros(n_scenes)
        scene_avg_est_depth = np.zeros(n_scenes)
        for i, scene_path in enumerate(scenes_paths):
            scene_name = scene_path.name
            # get RGB frames loader
            frames_loader = SceneLoader(
                scene_path=scene_path,
            )
            # examine the ground-truth depth maps
            gt_depth_loader = DepthAndEgoMotionLoader(
                scene_path=scene_path,
                depth_maps_source="ground_truth",
                egomotions_source="ground_truth",
            )
            scene_avg_gt_depth[i] = examine_depths(
                depth_loader=gt_depth_loader,
                frames_loader=frames_loader,
                save_path=save_path,
                scene_name=scene_name,
                depth_source="gt",
            )

            # examine the estimated depth maps
            est_depth_loader = DepthAndEgoMotionLoader(
                scene_path=scene_path,
                depth_maps_source="online_estimates",
                egomotions_source="online_estimates",
            )
            scene_avg_est_depth[i] = examine_depths(
                depth_loader=est_depth_loader,
                frames_loader=frames_loader,
                save_path=save_path,
                scene_name=scene_name,
                depth_source="est",
            )

        avg_gt_depth = np.mean(scene_avg_gt_depth)
        avg_est_depth = np.mean(scene_avg_est_depth)
        print(f"avg_gt_depth = {avg_gt_depth}")
        print(f"avg_est_depth = {avg_est_depth}")
        print(f"avg_gt_depth / avg_est_depth = {avg_gt_depth / avg_est_depth}")


# ---------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
