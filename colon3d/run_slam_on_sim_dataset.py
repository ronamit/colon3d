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
        default="data/sim_data/SimData9_with_tracks",
        help="Path to the dataset of scenes.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="data/sim_data/SimData9_with_tracks/Results",
        help="path to the save outputs",
    )
    parser.add_argument(
        "--depth_maps_source",
        type=str,
        default="online_estimates",
        choices=["ground_truth", "loaded_estimates", "online_estimates", "none"],
        help="The source of the depth maps, if 'ground_truth' then the ground truth depth maps will be loaded, "
        "if 'online_estimates' then the depth maps will be estimated online by the algorithm"
        "if 'loaded_estimates' then the depth maps estimations will be loaded, "
        "if 'none' then no depth maps will not be used,",
    )
    parser.add_argument(
        "--egomotions_source",
        type=str,
        default="online_estimates",
        choices=["ground_truth", "loaded_estimates", "online_estimates", "none"],
        help="The source of the egomotion, if 'ground_truth' then the ground truth egomotion will be loaded, "
        "if 'online_estimates' then the egomotion will be estimated online by the algorithm"
        "if 'loaded_estimates' then the egomotion estimations will be loaded, "
        "if 'none' then no egomotion will not be used,",
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
        "--n_examples_lim",
        type=int,
        default=0,
        help="upper limit on the number of examples used, if 0 then all examples are used",
    )

    args = parser.parse_args()
    dataset_path = Path(args.dataset_path).expanduser()
    assert dataset_path.exists(), f"dataset_path={dataset_path} does not exist"
    base_save_path = Path(args.save_path).expanduser()
    create_empty_folder(base_save_path, ask_overwrite=False)
    print(f"Outputs will be saved to {base_save_path}")
    examples_paths = list(dataset_path.glob("Scene_*"))
    examples_paths.sort()
    if args.n_examples_lim:
        examples_paths = examples_paths[: args.n_examples_lim]
    n_examples = len(examples_paths)
    metrics_table = pd.DataFrame()
    with Tee(base_save_path / "log_run_slam.txt"):  # save the prints to a file
        for i_example, example_path in enumerate(examples_paths):
            save_path = base_save_path / example_path.name
            print(
                "-" * 100
                + f"\nTime: {get_time_now_str()}\nRunning SLAM on example {i_example + 1} out of {n_examples}, results will be saved to {save_path}...\n"
                + "-" * 100,
            )
            save_path = Path(args.save_path).expanduser() / example_path.name
            create_empty_folder(save_path, ask_overwrite=True)

            _, metrics_stats = run_slam_on_scene(
                scene_path=example_path,
                save_path=save_path,
                n_frames_lim=args.n_frames_lim,
                alg_fov_ratio=args.alg_fov_ratio,
                depth_maps_source=args.depth_maps_source,
                egomotions_source=args.egomotions_source,
                save_all_plots=False,
                save_aided_nav_plot=True,
            )

            # add current example to the error metrics table
            if i_example == 0:
                metrics_table = pd.DataFrame(metrics_stats, index=[0])
            else:
                metrics_table.loc[i_example] = metrics_stats

            print("-" * 100)

    # save the error metrics table to a csv file
    metrics_table.to_csv(base_save_path / "err_table.csv", index=[0])
    print(f"Error metrics table saved to {base_save_path / 'err_table.csv'}")

    # compute statistics over all examples
    numeric_columns = metrics_table.select_dtypes(include=[np.number]).columns
    metrics_summary = {}
    for col in numeric_columns:
        mean_val = metrics_table[col].mean()
        std_val = metrics_table[col].std()
        n_examples = len(metrics_table[col])
        confidence_interval = 1.96 * std_val / np.sqrt(n_examples)  # 95% confidence interval
        metrics_summary[col] = f"{mean_val:.4f} +- {confidence_interval:.4f}"

    print("-" * 100 + "\nError metrics summary (mean +- 95\\% confidence interval):\n", metrics_summary)


# ---------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
