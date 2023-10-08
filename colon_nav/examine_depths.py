import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from colon_nav.alg.monocular_est_loader import DepthAndEgoMotionLoader
from colon_nav.util.data_util import SceneLoader, get_all_scenes_paths_in_dir
from colon_nav.util.general_util import (
    ArgsHelpFormatter,
    Tee,
    bool_arg,
    create_empty_folder,
    save_current_figure_and_close,
)
from colon_nav.util.torch_util import resize_single_image, to_numpy

# ---------------------------------------------------------------------------------------------------------------------
# plot for each example - the first frame ground truth and estimated of depth maps
# and calculate the mean depth the value of the estimated depth map and of the ground truth depth map
# ---------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(formatter_class=ArgsHelpFormatter)
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/sim_data/TrainData",
        help="Path to the dataset of scenes.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="data/models/MonoDepth2_orig/examination_result",
        help="Path to save the results.",
    )
    parser.add_argument(
        "-model_path",
        type=str,
        default="data_gcp/models/EndoSFM_tuned_v2",
        help="path to the saved depth and egomotion model (PoseNet and DepthNet) to be used for the case of online estimation",
    )
    parser.add_argument(
        "--n_scenes_lim",
        type=int,
        default=0,
        help="The number of scenes to examine, if 0 then all the scenes will be examined",
    )
    parser.add_argument(
        "--save_overwrite",
        type=bool_arg,
        default=True,
        help="If True then the results will be saved in the save_path folder, otherwise a new folder will be created",
    )
    args = parser.parse_args()
    print(f"args={args}")
    depth_examiner = DepthExaminer(
        dataset_path=Path(args.dataset_path),
        model_path=Path(args.model_path),
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
        model_path: Path,
        save_path: Path,
        depth_calib_method: str = "none",
        n_scenes_lim: int = 0,
        n_frames_lim: int = 0,
        save_overwrite: bool = True,
        frame_idx_to_show: int = 0,
    ):
        self.dataset_path = Path(dataset_path)
        self.model_path = Path(model_path)
        self.save_path = Path(save_path)
        self.depth_calib_method = depth_calib_method
        self.n_scenes_lim = n_scenes_lim
        self.n_frames_lim = n_frames_lim
        self.save_overwrite = save_overwrite
        self.frame_idx_to_show = frame_idx_to_show

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
            scenes_paths = get_all_scenes_paths_in_dir(dataset_path=self.dataset_path, with_targets=False)
            n_scenes = len(scenes_paths)
            n_scenes = min(n_scenes, self.n_scenes_lim) if self.n_scenes_lim > 0 else n_scenes
            print(f"n_scenes = {n_scenes}")
            scenes_paths.sort()

            # initialize the variables for the depth statistics (over all pixels in all frames in all scenes)
            n_pix_tot = 0
            sum_gt_depth = 0
            sum_est_depth = 0
            sum_est_depth_sqr = 0
            sum_gt_depth_times_est_depth = 0

            for i_scene in range(n_scenes):
                scene_path = scenes_paths[i_scene]
                scene_name = scene_path.name
                # get frames loader for current scene
                scene_loader = SceneLoader(
                    scene_path=scene_path,
                    n_frames_lim=self.n_frames_lim,
                )
                # get the ground truth depth loader
                gt_depth_loader = DepthAndEgoMotionLoader(
                    scene_path=scene_path,
                    scene_loader=scene_loader,
                    depth_maps_source="ground_truth",
                    egomotions_source="ground_truth",
                    model_path=None,
                )
                # get the estimated depth loader
                est_depth_loader = DepthAndEgoMotionLoader(
                    scene_path=scene_path,
                    scene_loader=scene_loader,
                    depth_maps_source="online_estimates",
                    egomotions_source="online_estimates",
                    model_path=self.model_path,
                )
                # calculate the depth statistics over all the frames in the scene
                n_frames = scene_loader.n_frames
                for i_frame in range(n_frames):
                    # Compute GT and estimated depth maps
                    gt_depths = compute_depths(
                        depth_loader=gt_depth_loader,
                        scene_loader=scene_loader,
                        frame_idx=i_frame,
                        save_path=self.save_path,
                        scene_name=scene_name,
                        fig_label="gt",
                        make_plots=(i_frame == self.frame_idx_to_show),
                    )
                    est_depths = compute_depths(
                        depth_loader=est_depth_loader,
                        scene_loader=scene_loader,
                        frame_idx=i_frame,
                        save_path=self.save_path,
                        scene_name=scene_name,
                        fig_label="est",
                        make_plots=(i_frame == self.frame_idx_to_show),
                    )
                    # flatten the depth maps
                    gt_depths = gt_depths.flatten()
                    est_depths = est_depths.flatten()
                    n_pix_new = len(gt_depths)
                    # update the depth statistics
                    n_pix_tot += n_pix_new
                    sum_gt_depth += np.sum(gt_depths)
                    sum_est_depth += np.sum(est_depths)
                    sum_gt_depth_times_est_depth += np.sum(gt_depths * est_depths)
                    sum_est_depth_sqr += np.sum(est_depths**2)

            avg_gt_depth = sum_gt_depth / n_pix_tot
            avg_est_depth = sum_est_depth / n_pix_tot
            avg_est_depth_sqr = sum_est_depth_sqr / n_pix_tot
            # empirical variance of the estimated depth
            var_est_depth = avg_est_depth_sqr - avg_est_depth**2
            # empirical covariance of the estimated depth and the ground truth depth
            cov_gt_depth_est_depth = sum_gt_depth_times_est_depth / n_pix_tot - avg_gt_depth * avg_est_depth
            # calculate the depth calibration parameters:
            if self.depth_calib_method == "none":
                depth_calib = {"depth_calib_type": "none", "depth_calib_a": 1, "depth_calib_b": 0}
            elif self.depth_calib_method == "linear":
                # linear model: depth = a * net_output, where a is the calibration parameter (calculated with least squares)
                depth_calib = {
                    "depth_calib_type": "linear",
                    "depth_calib_a": cov_gt_depth_est_depth / var_est_depth,
                    "depth_calib_b": 0,
                }
            elif self.depth_calib_method == "affine":
                # affine model: depth = a * net_output + b, where a and b are the calibration parameters (calculated with least squares)
                a = cov_gt_depth_est_depth / var_est_depth
                b = avg_gt_depth - a * avg_est_depth
                depth_calib = {"depth_calib_type": "affine", "depth_calib_a": a, "depth_calib_b": b}
            else:
                raise ValueError(f"Unknown depth calibration method: {self.depth_calib_method}")

            return depth_calib


# ---------------------------------------------------------------------------------------------------------------------


def compute_depths(
    depth_loader: DepthAndEgoMotionLoader,
    scene_loader: SceneLoader,
    frame_idx: int,
    save_path: Path,
    scene_name: str,
    fig_label: str,
    make_plots: bool = False,
):
    rgb_frame = scene_loader.get_frame_at_index(frame_idx=frame_idx)
    depth_map = depth_loader.get_depth_map_at_frame(frame_idx=frame_idx, rgb_frame=rgb_frame)

    if make_plots:
        # resize to the original image size
        depth_map_resized = resize_single_image(
            img=depth_map,
            new_height=rgb_frame.shape[0],
            new_width=rgb_frame.shape[1],
        )
        # resize to the original image size
        depth_map = resize_single_image(
            img=depth_map,
            new_height=rgb_frame.shape[0],
            new_width=rgb_frame.shape[1],
        )
        depth_map_resized = to_numpy(depth_map_resized, num_type="float_m")
        plt.figure()
        plt.imshow(depth_map_resized)
        plt.colorbar()
        fig_name = f"{scene_name}_depth_{fig_label}_{frame_idx}"
        plt.title(fig_name)
        plt.tight_layout()
        save_current_figure_and_close(save_path / fig_name)

        plt.figure()
        plt.imshow(rgb_frame)
        fig_name = f"{scene_name}_rgb_{frame_idx}"
        plt.title(fig_name)
        save_current_figure_and_close(save_path / fig_name)

    depth_map = to_numpy(depth_map, num_type="float_m")
    return depth_map


# ---------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------------------------------------------------
