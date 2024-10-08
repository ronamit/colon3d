import argparse
import os
from pathlib import Path

import attrs
import pandas as pd

from colon_nav.run_on_sim_scene import run_slam_on_scene
from colon_nav.util.data_util import get_all_scenes_paths_in_dir, get_full_scene_name
from colon_nav.util.general_util import (
    ArgsHelpFormatter,
    Tee,
    bool_arg,
    compute_metrics_statistics,
    create_empty_folder,
    get_time_now_str,
    to_path,
)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# ---------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(formatter_class=ArgsHelpFormatter)
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/datasets/ColonNav/Test",
        help="Path to the dataset of scenes.",
    )
    parser.add_argument(
        "--load_scenes_with_targets",
        type=bool_arg,
        default=True,
        help="If True, then only the scenes with targets under dataset_path will be loaded (use for datasets that have targets)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="data/results/ColonNav/temp_on_sim_dataset",
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
        "--model_path",
        type=str,
        default="data/models/EndoSFM_orig",
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
        load_scenes_with_targets=args.load_scenes_with_targets,
        save_raw_outputs=args.save_raw_outputs,
        depth_maps_source=args.depth_maps_source,
        egomotions_source=args.egomotions_source,
        model_path=Path(args.model_path),
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
    load_scenes_with_targets: bool = True
    save_raw_outputs: bool = False
    depth_maps_source: str = "none"
    egomotions_source: str = "none"
    model_path: str | Path | None = None
    alg_fov_ratio: float = 0
    n_frames_lim: int = 0
    n_scenes_lim: int = 0
    save_overwrite: bool = True
    alg_settings_override: dict | None = None
    print_interval: int = 20

    # ---------------------------------------------------------------------------------------------------------------------

    def run(self):
        self.dataset_path = Path(self.dataset_path)
        self.save_path = Path(self.save_path)
        assert self.dataset_path.exists(), f"dataset_path={self.dataset_path} does not exist"
        if self.save_path.exists():
            print(f"save_path={self.save_path} already exists...")
        if self.save_overwrite:
            print("Removing the existing results...")
            create_empty_folder(self.save_path, save_overwrite=True)
        # if the run was already completed then skip it
        if (self.save_path / "metrics_summary.csv").exists():
            print("Run was already completed, skipping it...")
            metrics_stats = pd.read_csv(self.save_path / "metrics_summary.csv").to_dict(orient="records")[0]
            return metrics_stats

        print(f"Outputs will be saved to {self.save_path}")
        scenes_paths = get_all_scenes_paths_in_dir(
            dataset_path=self.dataset_path,
            with_targets=self.load_scenes_with_targets,
        )

        assert len(scenes_paths) > 0, f"No scenes found in {self.dataset_path}"

        if self.n_scenes_lim:
            scenes_paths = scenes_paths[: self.n_scenes_lim]
        n_scenes = len(scenes_paths)

        with Tee(self.save_path / "log_run_slam.txt"):  # save the prints to a file
            print(f"Running SLAM on {n_scenes} scenes from the dataset {self.dataset_path}...")

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # function to run the SLAM algorithm per scene
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            def run_on_scene(i_scene: int) -> dict:
                scene_path = scenes_paths[i_scene]
                scene_name = get_full_scene_name(scene_path)
                scene_save_path = self.save_path / scene_name
                # check if the scene was already completed
                if (scene_save_path / "metrics_stats.csv").exists():
                    print(f"Scene {i_scene + 1} out of {n_scenes} was already completed, skipping it...")
                    metrics_stats = pd.read_csv(scene_save_path / "metrics_stats.csv").to_dict(orient="records")[0]
                    return metrics_stats
                print(
                    "-" * 100
                    + f"\nTime: {get_time_now_str()}\nRunning SLAM on scene {i_scene + 1} out of {n_scenes}, results will be saved to {scene_save_path}...\n"
                    + "-" * 100,
                )

                # create the save folder
                create_empty_folder(scene_save_path, save_overwrite=True)

                # run the SLAM algorithm on the current scene
                _, scene_metrics_stats = run_slam_on_scene(
                    scene_path=scene_path,
                    save_path=scene_save_path,
                    save_raw_outputs_path=self.save_raw_outputs,
                    n_frames_lim=self.n_frames_lim,
                    alg_fov_ratio=self.alg_fov_ratio,
                    depth_maps_source=self.depth_maps_source,
                    egomotions_source=self.egomotions_source,
                    model_path=to_path(self.model_path),
                    alg_settings_override=self.alg_settings_override,
                    example_name=scene_name,
                    print_interval=self.print_interval,
                )
                print("-" * 20 + f"\nFinished running SLAM on scene {i_scene + 1} out of {n_scenes}\n" + "-" * 20)
                return scene_metrics_stats

            # ------------------------------------------------------------------------

            # run the SLAM algorithm on all scenes
            all_scene_metrics_stats = []
            for i_scene in range(n_scenes):
                scene_metrics = run_on_scene(i_scene)
                all_scene_metrics_stats.append(scene_metrics)

            metrics_table = pd.DataFrame()
            for i_scene in range(n_scenes):
                metrics_stats = all_scene_metrics_stats[i_scene]
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
            metrics_summary = compute_metrics_statistics(metrics_table)
            print("-" * 100 + "\nError metrics summary (mean +- 95\\% confidence interval):\n", metrics_summary)
            # save to csv file
            metrics_summary = {"run_name": self.save_path.name} | metrics_summary
            metrics_summary_df = pd.DataFrame(metrics_summary, index=["run_name"])
            metrics_summary_df.to_csv(self.save_path / "metrics_summary.csv", index=False)
            print("-" * 100)
            return None


# ---------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------------------------------------------------
