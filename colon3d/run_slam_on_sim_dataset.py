import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from colon3d.run_slam_on_sim_scene import run_slam_on_scene
from colon3d.utils.general_util import ArgsHelpFormatter, Tee, create_empty_folder, get_time_now_str

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

    args = parser.parse_args()
    dataset_path = Path(args.dataset_path).expanduser()
    assert dataset_path.exists(), f"dataset_path={dataset_path} does not exist"
    base_save_path = Path(args.save_path).expanduser()
    create_empty_folder(base_save_path, ask_overwrite=False)
    print(f"Outputs will be saved to {base_save_path}")
    cases_paths = list(dataset_path.glob("Scene_*"))
    cases_paths.sort()
    if args.n_cases_lim:
        cases_paths = cases_paths[: args.n_cases_lim]
    n_cases = len(cases_paths)
    metrics_table = pd.DataFrame()
    with Tee(base_save_path / "log_run_slam.txt"):  # save the prints to a file
        for i_case, case_path in enumerate(cases_paths):
            save_path = base_save_path / case_path.name
            print(
                "-" * 100
                + f"\nTime: {get_time_now_str()}\nRunning SLAM on case {i_case + 1} out of {n_cases}, results will be saved to {save_path}...\n"
                + "-" * 100,
            )
            save_path = Path(args.save_path).expanduser() / case_path.name
            create_empty_folder(save_path, ask_overwrite=True)

            # run the SLAM algorithm on the current case
            _, metrics_stats = run_slam_on_scene(
                scene_path=case_path,
                save_path=save_path,
                n_frames_lim=args.n_frames_lim,
                alg_fov_ratio=args.alg_fov_ratio,
                depth_maps_source=args.depth_maps_source,
                egomotions_source=args.egomotions_source,
                depth_and_egomotion_model_path=args.depth_and_egomotion_model_path,
                plot_names=["aided_nav", "keypoints_and_tracks"], # plots to create
            )

            # add current case to the error metrics table
            if i_case == 0:
                metrics_table = pd.DataFrame(metrics_stats, index=[0])
            else:
                metrics_table.loc[i_case] = metrics_stats

            print("-" * 100)

    # save the error metrics table to a csv file
    metrics_table.to_csv(base_save_path / "err_table.csv", index=[0])
    print(f"Error metrics table saved to {base_save_path / 'err_table.csv'}")

    # compute statistics over all cases
    numeric_columns = metrics_table.select_dtypes(include=[np.number]).columns
    metrics_summary = {}
    for col in numeric_columns:
        mean_val = np.nanmean(metrics_table[col]) # ignore nan values
        std_val = np.nanstd(metrics_table[col])
        n_cases = np.sum(~np.isnan(metrics_table[col]))
        confidence_interval = 1.96 * std_val / np.sqrt(max(n_cases, 1))  # 95% confidence interval
        metrics_summary[col] = f"{mean_val:.4f} +- {confidence_interval:.4f}"

    print("-" * 100 + "\nError metrics summary (mean +- 95\\% confidence interval):\n", metrics_summary)
    # save to csv file
    pd.DataFrame(metrics_summary, index=[0]).to_csv(base_save_path / "metrics_summary.csv", index=[0])


# ---------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
