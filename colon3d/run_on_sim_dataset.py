import argparse
from pathlib import Path

import attrs
import numpy as np
import pandas as pd

from colon3d.run_on_sim_scene import run_slam_on_scene
from colon3d.util.general_util import (
    ArgsHelpFormatter,
    Tee,
    bool_arg,
    create_empty_folder,
    get_time_now_str,
)

# ---------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(formatter_class=ArgsHelpFormatter)
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/sim_data/TestData21_cases",
        help="Path to the dataset of scenes.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="results/Temp/temp_on_sim_dataset",
        help="path to the save outputs",
    )
    parser.add_argument(
        "--save_raw_outputs",
        type=bool_arg,
        default=False,
        help="If True then all the raw outputs will be saved (as pickle file), not just the plots and summary.",
    )
    parser.add_argument(
        "--depth_maps_source",
        type=str,
        default="ground_truth",
        choices=["ground_truth", "loaded_estimates", "online_estimates", "none"],
        help="The source of the depth maps, if 'ground_truth' then the ground truth depth maps will be loaded, "
        "if 'online_estimates' then the depth maps will be estimated online by the algorithm"
        "if 'loaded_estimates' then the depth maps estimations will be loaded, "
        "if 'none' then no depth maps will not be used,",
    )
    parser.add_argument(
        "--egomotions_source",
        type=str,
        default="ground_truth",
        choices=["ground_truth", "loaded_estimates", "online_estimates", "none"],
        help="The source of the egomotion, if 'ground_truth' then the ground truth egomotion will be loaded, "
        "if 'online_estimates' then the egomotion will be estimated online by the algorithm"
        "if 'loaded_estimates' then the egomotion estimations will be loaded, "
        "if 'none' then no egomotion will not be used,",
    )
    parser.add_argument(
        "--depth_and_egomotion_method",
        type=str,
        default="EndoSFM",
        choices=["EndoSFM", "MonoDepth2", "SC-DepthV3"],
        help="The method used for depth and egomotion estimation (to be used for the case of online estimation))",
    )
    parser.add_argument(
        "--depth_and_egomotion_model_path",
        type=str,
        default="saved_models/EndoSFM_orig",
        help="path to the saved depth and egomotion model (PoseNet and DepthNet) to be used for the case of online estimation",
    )
    parser.add_argument(
        "--alg_fov_ratio",
        type=float,
        default=0,
        help="If in range (0,1) then the algorithm will use only a fraction of the frames, if 0 then all of the frame is used.",
    )
    parser.add_argument(
        "--n_frames_lim",
        type=int,
        default=0,
        help="upper limit on the number of frames used, if 0 then all frames are used",
    )
    parser.add_argument(
        "--n_scenes_lim",
        type=int,
        default=1,
        help="upper limit on the number of scenes used, if 0 then all scenes are used",
    )
    parser.add_argument(
        "--save_overwrite",
        type=bool_arg,
        default=True,
        help="If True then the save folder will be overwritten if it already exists",
    )

    args = parser.parse_args()
    print(f"args={args}")
    slam_on_dataset_runner = SlamOnDatasetRunner(
        dataset_path=Path(args.dataset_path),
        save_path=Path(args.save_path),
        save_raw_outputs=args.save_raw_outputs,
        depth_maps_source=args.depth_maps_source,
        egomotions_source=args.egomotions_source,
        depth_and_egomotion_model_path=Path(args.depth_and_egomotion_model_path),
        alg_fov_ratio=args.alg_fov_ratio,
        n_frames_lim=args.n_frames_lim,
        n_scenes_lim=args.n_scenes_lim,
        save_overwrite=args.save_overwrite,
    )

    slam_on_dataset_runner.run()


# ---------------------------------------------------------------------------------------------------------------------


@attrs.define
class SlamOnDatasetRunner:
    dataset_path: Path
    save_path: Path
    save_raw_outputs: bool = False
    depth_maps_source: str = "none"
    egomotions_source: str = "none"
    depth_and_egomotion_method: str | None = None
    depth_and_egomotion_model_path: Path | None = None
    alg_fov_ratio: float = 0
    n_frames_lim: int = 0
    n_scenes_lim: int = 0
    save_overwrite: bool = True
    plot_aided_nav: bool = True
    alg_settings_override: dict | None = None
    

    # ---------------------------------------------------------------------------------------------------------------------

    def run(self):
        self.dataset_path = Path(self.dataset_path)
        self.save_path = Path(self.save_path)
        assert self.dataset_path.exists(), f"dataset_path={self.dataset_path} does not exist"
        is_created = create_empty_folder(self.save_path, save_overwrite=self.save_overwrite)
        if not is_created:
            print(f"{self.save_path} already exists.. " + "-" * 50)
            return
        print(f"Outputs will be saved to {self.save_path}")
        scenes_paths = list(self.dataset_path.glob("Scene_*"))
        scenes_paths.sort()
        if self.n_scenes_lim:
            scenes_paths = scenes_paths[: self.n_scenes_lim]
        n_scenes = len(scenes_paths)

        with Tee(self.save_path / "log_run_slam.txt"):  # save the prints to a file
            print(f"Running SLAM on {n_scenes} scenes from the dataset {self.dataset_path}...")

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # function to run the SLAM algorithm per scene
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            def run_on_scene(i_scene: int):
                scene_path = scenes_paths[i_scene]
                save_path = self.save_path / scene_path.name
                print(
                    "-" * 100
                    + f"\nTime: {get_time_now_str()}\nRunning SLAM on scene {i_scene + 1} out of {n_scenes}, results will be saved to {save_path}...\n"
                    + "-" * 100,
                )
                save_path = Path(self.save_path) / scene_path.name
                create_empty_folder(save_path, save_overwrite=True)
                
                # result plots to save
                plot_names = ["aided_nav", "keypoints_and_tracks"] if self.plot_aided_nav else ["keypoints_and_tracks"]
                
                # run the SLAM algorithm on the current scene
                _, metrics_stats = run_slam_on_scene(
                    scene_path=scene_path,
                    save_path=save_path,
                    save_raw_outputs=self.save_raw_outputs,
                    n_frames_lim=self.n_frames_lim,
                    alg_fov_ratio=self.alg_fov_ratio,
                    depth_maps_source=self.depth_maps_source,
                    egomotions_source=self.egomotions_source,
                    depth_and_egomotion_method=self.depth_and_egomotion_method,
                    depth_and_egomotion_model_path=Path(self.depth_and_egomotion_model_path),
                    alg_settings_override=self.alg_settings_override,
                    plot_names=plot_names,  # plots to create
                )
                print("-" * 20 + f"\nFinished running SLAM on scene {i_scene + 1} out of {n_scenes}\n" + "-" * 20)
                return metrics_stats

            # ------------------------------------------------------------------------

            metrics_stats_all = []
            for i_scene in range(n_scenes):
                metrics_stats_all.append(run_on_scene(i_scene))

            metrics_table = pd.DataFrame()
            for i_scene in range(n_scenes):
                metrics_stats = metrics_stats_all[i_scene]
                # add current scene to the error metrics table
                if i_scene == 0:
                    metrics_table = pd.DataFrame(metrics_stats, index=[0])
                else:
                    metrics_table.loc[i_scene] = metrics_stats

            print(f"Finished running SLAM on {n_scenes} scenes from the dataset {self.dataset_path}...")
            # save the error metrics table to a csv file
            metrics_table.to_csv(self.save_path / "err_table.csv", index=[0])
            print(f"Error metrics table saved to {self.save_path / 'err_table.csv'}")

            # compute statistics over all scenes
            numeric_columns = metrics_table.select_dtypes(include=[np.number]).columns
            metrics_summary = {}
            for col in numeric_columns:
                mean_val = np.nanmean(metrics_table[col])  # ignore nan values
                std_val = np.nanstd(metrics_table[col])
                n_scenes = np.sum(~np.isnan(metrics_table[col]))
                confidence_interval = 1.96 * std_val / np.sqrt(max(n_scenes, 1))  # 95% confidence interval
                metrics_summary[col] = f"{mean_val:.4f} +- {confidence_interval:.4f}"

            print("-" * 100 + "\nError metrics summary (mean +- 95\\% confidence interval):\n", metrics_summary)
            # save to csv file
            metrics_summary = {"save_path": str(self.save_path), "run_name": self.save_path.name} | metrics_summary
            pd.DataFrame(metrics_summary, index=[0]).to_csv(self.save_path / "metrics_summary.csv", index=[0])
            print("-" * 100)


# ---------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------------------------------------------------
