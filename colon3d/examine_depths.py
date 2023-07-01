import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from colon3d.utils.data_util import SceneLoader
from colon3d.utils.depth_egomotion import DepthAndEgoMotionLoader
from colon3d.utils.general_util import ArgsHelpFormatter, Tee, create_empty_folder, save_plot_and_close
from colon3d.utils.torch_util import to_numpy

# ---------------------------------------------------------------------------------------------------------------------
# plot for each example - the first frame ground truth and estimated of depth maps
# and calculate the mean depth the value of the estimated depth map and of the ground truth depth map
# ---------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(formatter_class=ArgsHelpFormatter)
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/sim_data/TrainData22",
        help="Path to the dataset of scenes.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="saved_models/EndoSFM_tuned/DepthExaminerResults",
        help="Path to save the results.",
    )
    parser.add_argument(
        "--depth_and_egomotion_model_path",
        type=str,
        default="saved_models/EndoSFM_tuned",
        help="path to the saved depth and egomotion model (PoseNet and DepthNet) to be used for online estimation",
    )
    parser.add_argument(
        "--n_scenes_lim",
        type=int,
        default=0,
        help="The number of scenes to examine, if 0 then all the scenes will be examined",
    )
    parser.add_argument(
        "--save_overwrite",
        type=bool,
        default=True,
        help="If True then the results will be saved in the save_path folder, otherwise a new folder will be created",
    )
    args = parser.parse_args()
    depth_examiner = DepthExaminer(
        dataset_path=Path(args.dataset_path),
        depth_and_egomotion_model_path=Path(args.depth_and_egomotion_model_path),
        save_path=Path(args.save_path),
        n_scenes_lim=args.n_scenes_lim,
        save_overwrite=args.save_overwrite,
    )

    depth_examiner.run()
    
# ---------------------------------------------------------------------------------------------------------------------


class DepthExaminer:
    def __init__(
        self,
        dataset_path: Path,
        depth_and_egomotion_model_path: Path,
        save_path: Path,
        n_scenes_lim: int = 0,
        save_overwrite: bool = True,
    ):
        self.dataset_path = Path(dataset_path)
        self.depth_and_egomotion_model_path = Path(depth_and_egomotion_model_path)
        self.save_path = Path(save_path)
        self.n_scenes_lim = n_scenes_lim
        self.save_overwrite = save_overwrite
        
    # ---------------------------------------------------------------------------------------------------------------------

    def run(self):
        # check if the results already exist - if so then load them and return
        results_file_path = self.save_path / "depth_exam.yaml"
        if results_file_path.exists() and not self.save_overwrite:
            print(f"{self.save_path} already exists...\n" + "-" * 50)
            results = yaml.load(results_file_path.open("r"), Loader=yaml.FullLoader)
            return results
        
        create_empty_folder(self.save_path, save_overwrite=True)
        
        with Tee(self.save_path / "examine_depths.log"):
            scenes_paths = list(self.dataset_path.glob("Scene_*"))
            n_scenes = len(scenes_paths)
            n_scenes = min(n_scenes, self.n_scenes_lim) if self.n_scenes_lim > 0 else n_scenes
            print(f"n_scenes = {n_scenes}")
            scenes_paths.sort()
            scene_avg_gt_depth = np.zeros(n_scenes)
            scene_avg_est_depth = np.zeros(n_scenes)
            for i_scene in range(n_scenes):
                scene_path = scenes_paths[i_scene]
                scene_name = scene_path.name
                # get RGB frames loader
                scene_loader = SceneLoader(
                    scene_path=scene_path,
                )
                # examine the ground-truth depth maps
                gt_depth_loader = DepthAndEgoMotionLoader(
                    scene_path=scene_path,
                    depth_maps_source="ground_truth",
                    egomotions_source="ground_truth",
                    depth_and_egomotion_model_path=None,
                )
                scene_avg_gt_depth[i_scene] = compute_depths(
                    depth_loader=gt_depth_loader,
                    scene_loader=scene_loader,
                    save_path=self.save_path,
                    scene_name=scene_name,
                    fig_label="gt",
                )

                # examine the estimated depth maps
                est_depth_loader = DepthAndEgoMotionLoader(
                    scene_path=scene_path,
                    depth_maps_source="online_estimates",
                    egomotions_source="online_estimates",
                    depth_and_egomotion_model_path=self.depth_and_egomotion_model_path,
                )
                scene_avg_est_depth[i_scene] = compute_depths(
                    depth_loader=est_depth_loader,
                    scene_loader=scene_loader,
                    save_path=self.save_path,
                    scene_name=scene_name,
                    fig_label="est",
                )

            avg_gt_depth = np.mean(scene_avg_gt_depth)
            std_gt_depth = np.std(scene_avg_gt_depth)
            avg_est_depth = np.mean(scene_avg_est_depth)
            std_est_depth = np.std(scene_avg_est_depth)
            # save the results as yaml file
            with (self.save_path / "depth_exam.yaml").open("w") as f:
                depth_exam = {
                    "dataset_path": self.dataset_path,
                    "avg_gt_depth": avg_gt_depth,
                    "avg_est_depth": avg_est_depth,
                    "std_gt_depth": std_gt_depth,
                    "std_est_depth": std_est_depth,
                    "avg_gt_depth / avg_est_depth": avg_gt_depth / avg_est_depth,
                    "std_gt_depth / std_est_depth": std_gt_depth / std_est_depth,
                }
                yaml.dump(depth_exam, f)
            print(depth_exam)
            return depth_exam


# ---------------------------------------------------------------------------------------------------------------------


def compute_depths(
    depth_loader: DepthAndEgoMotionLoader,
    scene_loader: SceneLoader,
    save_path: Path,
    scene_name: str,
    fig_label: str,
):
    # save a heatmap of the first frame depth map
    # and RGB image of the first frame
    frame_idx = 0
    rgb_frame = scene_loader.get_frame_at_index(frame_idx=frame_idx)
    depth_map = depth_loader.get_depth_map_at_frame(frame_idx=frame_idx, rgb_frame=rgb_frame)
    depth_map = to_numpy(depth_map)
    plt.figure()
    plt.imshow(depth_map)
    plt.colorbar()
    fig_name = f"{scene_name}_depth_{fig_label}_{frame_idx}"
    plt.title(fig_name)
    plt.tight_layout()
    save_plot_and_close(save_path / fig_name)

    plt.figure()
    plt.imshow(rgb_frame)
    fig_name = f"{scene_name}_rgb_{frame_idx}"
    plt.title(fig_name)
    save_plot_and_close(save_path / fig_name)

    # calculate the mean depth of all frames
    n_frames = scene_loader.n_frames
    all_frames_avg_depth = np.zeros(n_frames)
    for i in range(n_frames):
        rgb_frame = scene_loader.get_frame_at_index(frame_idx=i)
        depth_map = depth_loader.get_depth_map_at_frame(frame_idx=i, rgb_frame=rgb_frame)
        depth_map = to_numpy(depth_map)
        all_frames_avg_depth[i] = np.mean(depth_map)
    scene_avg_depth = np.mean(all_frames_avg_depth)
    return scene_avg_depth


# ---------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------------------------------------------------
