import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from colon3d.run_slam_on_sim_scene import run_slam_on_scene
from colon3d.utils.general_util import ArgsHelpFormatter, Tee, bool_arg, create_empty_folder, get_time_now_str

# ---------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(formatter_class=ArgsHelpFormatter)
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/sim_data/SimData14_test_cases",
        help="Path to the dataset of scenes.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="data/sim_data/SimData14_test_cases/Results",
        help="path to the save outputs",
    )
    parser.add_argument(
        "--depth_maps_source",
        type=str,
        default="none",
        choices=["ground_truth", "loaded_estimates", "online_estimates", "none"],
        help="The source of the depth maps, if 'ground_truth' then the ground truth depth maps will be loaded, "
        "if 'online_estimates' then the depth maps will be estimated online by the algorithm"
        "if 'loaded_estimates' then the depth maps estimations will be loaded, "
        "if 'none' then no depth maps will not be used,",
    )
    parser.add_argument(
        "--egomotions_source",
        type=str,
        default="none",
        choices=["ground_truth", "loaded_estimates", "online_estimates", "none"],
        help="The source of the egomotion, if 'ground_truth' then the ground truth egomotion will be loaded, "
        "if 'online_estimates' then the egomotion will be estimated online by the algorithm"
        "if 'loaded_estimates' then the egomotion estimations will be loaded, "
        "if 'none' then no egomotion will not be used,",
    )
    parser.add_argument(
        "--depth_and_egomotion_model_path",
        type=str,
        default="saved_models/EndoSFM_orig",
        help="path to the saved depth and egomotion model (PoseNet and DepthNet) to be used for online estimation",
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
        "--n_cases_lim",
        type=int,
        default=0,
        help="upper limit on the number of cases used, if 0 then all cases are used",
    )
    parser.add_argument(
        "--save_overwrite",
        type=bool_arg,
        default=True,
        help="If True then the save folder will be overwritten if it already exists",
    )

    args = parser.parse_args()

    slam_on_dataset_runner = SlamOnDatasetRunner(
        dataset_path=Path(args.dataset_path),
        save_path=Path(args.save_path),
        depth_maps_source=args.depth_maps_source,
        egomotions_source=args.egomotions_source,
        depth_and_egomotion_model_path=Path(args.depth_and_egomotion_model_path),
        alg_fov_ratio=args.alg_fov_ratio,
        n_frames_lim=args.n_frames_lim,
        n_cases_lim=args.n_cases_lim,
        save_overwrite=args.save_overwrite,
    )

    slam_on_dataset_runner.run()


# ---------------------------------------------------------------------------------------------------------------------


class SlamOnDatasetRunner:
    def __init__(
        self,
        dataset_path: Path,
        save_path: Path,
        depth_maps_source: str,
        egomotions_source: str,
        depth_and_egomotion_model_path: Path | None = None,
        alg_fov_ratio: float = 0,
        n_frames_lim: int = 0,
        n_cases_lim: int = 0,
        use_bundle_adjustment: bool = True,
        save_overwrite: bool = True,
    ):
        self.dataset_path = Path(dataset_path)
        assert dataset_path.exists(), f"dataset_path={dataset_path} does not exist"
        self.save_path = Path(save_path)
        self.depth_maps_source = depth_maps_source
        self.egomotions_source = egomotions_source
        self.depth_and_egomotion_model_path = depth_and_egomotion_model_path
        self.alg_fov_ratio = alg_fov_ratio
        self.n_frames_lim = n_frames_lim
        self.n_cases_lim = n_cases_lim
        self.save_overwrite = save_overwrite
        self.use_bundle_adjustment = use_bundle_adjustment

    # ---------------------------------------------------------------------------------------------------------------------

    def run(self):
        is_created = create_empty_folder(self.save_path, save_overwrite=self.save_overwrite)
        if not is_created:
            print(f"{self.save_path} already exists.. " + "-" * 50)
            return
        print(f"Outputs will be saved to {self.save_path}")
        cases_paths = list(self.dataset_path.glob("Scene_*"))
        cases_paths.sort()
        if self.n_cases_lim:
            cases_paths = cases_paths[: self.n_cases_lim]
        n_cases = len(cases_paths)
        metrics_table = pd.DataFrame()
        with Tee(self.save_path / "log_run_slam.txt"):  # save the prints to a file
            print(f"Running SLAM on {n_cases} cases from the dataset {self.dataset_path}...")
            for i_case, case_path in enumerate(cases_paths):
                save_path = self.save_path / case_path.name
                print(
                    "-" * 100
                    + f"\nTime: {get_time_now_str()}\nRunning SLAM on case {i_case + 1} out of {n_cases}, results will be saved to {save_path}...\n"
                    + "-" * 100,
                )
                save_path = Path(self.save_path) / case_path.name
                create_empty_folder(save_path, save_overwrite=True)
                # run the SLAM algorithm on the current case
                _, metrics_stats = run_slam_on_scene(
                    scene_path=case_path,
                    save_path=save_path,
                    n_frames_lim=self.n_frames_lim,
                    alg_fov_ratio=self.alg_fov_ratio,
                    depth_maps_source=self.depth_maps_source,
                    egomotions_source=self.egomotions_source,
                    depth_and_egomotion_model_path=self.depth_and_egomotion_model_path,
                    use_bundle_adjustment=self.use_bundle_adjustment,
                    plot_names=["aided_nav", "keypoints_and_tracks"],  # plots to create
                )
                # add current case to the error metrics table
                if i_case == 0:
                    metrics_table = pd.DataFrame(metrics_stats, index=[0])
                else:
                    metrics_table.loc[i_case] = metrics_stats
                print("-" * 100)
                
        print(f"Finished running SLAM on {n_cases} cases from the dataset {self.dataset_path}...")
        # save the error metrics table to a csv file
        metrics_table.to_csv(self.save_path / "err_table.csv", index=[0])
        print(f"Error metrics table saved to {self.save_path / 'err_table.csv'}")

        # compute statistics over all cases
        numeric_columns = metrics_table.select_dtypes(include=[np.number]).columns
        metrics_summary = {}
        for col in numeric_columns:
            mean_val = np.nanmean(metrics_table[col])  # ignore nan values
            std_val = np.nanstd(metrics_table[col])
            n_cases = np.sum(~np.isnan(metrics_table[col]))
            confidence_interval = 1.96 * std_val / np.sqrt(max(n_cases, 1))  # 95% confidence interval
            metrics_summary[col] = f"{mean_val:.4f} +- {confidence_interval:.4f}"

        print("-" * 100 + "\nError metrics summary (mean +- 95\\% confidence interval):\n", metrics_summary)
        # save to csv file
        metrics_summary = {"save_path": str(self.save_path)} | metrics_summary
        pd.DataFrame(metrics_summary, index=[0]).to_csv(self.save_path / "metrics_summary.csv", index=[0])
        print("-" * 100)


# ---------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------------------------------------------------
