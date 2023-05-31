import argparse
from pathlib import Path

from colon3d.general_util import Tee, create_empty_folder, get_time_now_str
from colon3d.run_slam_on_sim_example import run_slam_on_example

# ---------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/sim_data/SimData2/Examples",
        help="Path to the dataset of examples.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="data/sim_data/SimData2/Examples/Results",
        help="path to the save outputs",
    )
    parser.add_argument(
        "--alg_fov_ratio",
        type=float,
        default=0.99,
        help="The FOV ratio (in the range [0,1]) used for the SLAM algorithm, out of the original FOV, the rest is hidden and only used for validation",
    )
    parser.add_argument(
        "--n_frames_lim",
        type=int,
        default=0,
        help="upper limit on the number of frames used, if 0 then all frames are used",
    )

    args = parser.parse_args()
    save_path = Path(args.save_path).expanduser()
    dataset_path = Path(args.dataset_path).expanduser()
    create_empty_folder(save_path)
    print(f"Outputs will be saved to {save_path}")
    examples_paths = list(dataset_path.glob("Seq_*"))
    with Tee(save_path / "log_run_slam.txt"):  # save the prints to a file
        for i_example, example_path in enumerate(examples_paths):
            print(
                "-" * 100
                + f"\nTime: {get_time_now_str()}\nRunning SLAM on example {i_example + 1} out of {len(examples_paths)}...\n"
                + "-" * 100,
            )
            save_path = Path(args.save_path).expanduser() / example_path.name
            create_empty_folder(save_path)
            run_slam_on_example(
                example_path=example_path,
                save_path=save_path,
                n_frames_lim=args.n_frames_lim,
                alg_fov_ratio=args.alg_fov_ratio,
            )


# ---------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
