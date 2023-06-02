import argparse
import pickle
from pathlib import Path

from colon3d.alg_settings import AlgorithmParam
from colon3d.data_util import FramesLoader
from colon3d.depth_util import DepthAndEgoMotionLoader
from colon3d.general_util import Tee, create_empty_folder
from colon3d.show_slam_out import save_slam_out_plots
from colon3d.slam_alg import SlamRunner
from colon3d.tracks_util import DetectionsTracker

# ---------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--example_path",
        type=str,
        default="data/my_videos/Example_4",
        help=" path to the video",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="data/my_videos/Example_4/Short_Results",
        help="path to the save outputs",
    )
    parser.add_argument(
        "--alg_fov_ratio",
        type=float,
        default=0.8,
        help="The FOV ratio (in the range [0,1]) used for the SLAM algorithm, out of the original FOV, the rest is hidden and only used for validation",
    )
    parser.add_argument(
        "--n_frames_lim",
        type=int,
        default=50,
        help="upper limit on the number of frames used, if 0 then all frames are used",
    )
    parser.add_argument(
        "--draw_interval",
        type=int,
        default=50,
        help="plot and save figures each draw_interval frames",
    )

    args = parser.parse_args()
    save_path = Path(args.save_path).expanduser()
    create_empty_folder(save_path)
    print(f"Outputs will be saved to {save_path}")

    with Tee(save_path / "log_run_slam.txt"):  # save the prints to a file
        frames_loader = FramesLoader(
            sequence_path=args.example_path,
            n_frames_lim=args.n_frames_lim,
            alg_fov_ratio=args.alg_fov_ratio,
        )
        detections_tracker = DetectionsTracker(
            example_path=args.example_path,
            frames_loader=frames_loader,
        )
        depth_estimator = DepthAndEgoMotionLoader(
            example_path=args.example_path,
            source="estimated",
        )

        # get the default parameters for the SLAM algorithm
        alg_prm = AlgorithmParam()

        # Run the SLAM algorithm
        slam_runner = SlamRunner(alg_prm)
        slam_out = slam_runner.run(
            frames_loader=frames_loader,
            detections_tracker=detections_tracker,
            depth_estimator=depth_estimator,
            save_path=save_path,
            draw_interval=args.draw_interval,
        )

        if save_path:
            results_file_path = save_path / "out_variables.pkl"
            with results_file_path.open("wb") as file:
                pickle.dump(slam_out, file)
                print(f"Saved the results to {results_file_path}")
        # Show results
        save_slam_out_plots(slam_out=slam_out, save_path=args.save_path, example_path=args.example_path)


# ---------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
