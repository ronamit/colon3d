import argparse
import pickle
from pathlib import Path

from colon3d.alg_settings import AlgorithmParam
from colon3d.data_util import FramesLoader
from colon3d.depth_util import DepthAndEgoMotionLoader
from colon3d.detections_util import DetectionsTracker
from colon3d.general_util import Tee, create_empty_folder
from colon3d.show_slam_out import show_slam_out
from colon3d.slam_alg import SlamRunner

# ---------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--example_path",
        type=str,
        default="data/sim_data/SimData2/Examples/Seq_00000_0000",
        help=" path to the video",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="data/sim_data/SimData2/Examples/Seq_00000_0000/Results",
        help="path to the save outputs",
    )
    parser.add_argument(
        "--alg_fov_ratio",
        type=float,
        default=0.95,
        help="The FOV ratio (in the range [0,1]) used for the SLAM algorithm, out of the original FOV, the rest is hidden and only used for validation",
    )
    parser.add_argument(
        "--n_frames_lim",
        type=int,
        default=0,
        help="upper limit on the number of frames used, if 0 then all frames are used",
    )
    parser.add_argument(
        "--draw_interval",
        type=int,
        default=100,
        help="plot and save figures each draw_interval frames",
    )

    args = parser.parse_args()
    save_path = Path(args.save_path).expanduser()
    example_path = Path(args.example_path).expanduser()
    create_empty_folder(save_path)
    print(f"Outputs will be saved to {save_path}")

    with Tee(save_path / "log_run_slam.txt"):  # save the prints to a file
        frames_loader = FramesLoader(
            sequence_path=example_path,
            n_frames_lim=args.n_frames_lim,
            alg_fov_ratio=args.alg_fov_ratio,
        )
        detections_tracker = DetectionsTracker(
            example_path=example_path,
            frames_loader=frames_loader,
        )
        depth_estimator = DepthAndEgoMotionLoader(
            example_path=example_path,
            source="estimated",
        )

        # load the detections ground truth info
        with (example_path / "tracks_info.pkl").open("rb") as file:
            tracks_info = pickle.load(file)
        print("-" * 20, "\nGround truth tracks info: ", tracks_info)
        tracks_time = tracks_info["frame_inds"] / frames_loader.fps
        print(f"Frames Times :{tracks_time}[sec]\n", "-" * 20, "\n")

        # get the default parameters for the SLAM algorithm
        alg_prm = AlgorithmParam()

        # Run the SLAM algorithm
        slam_runner = SlamRunner(alg_prm)
        slam_out = slam_runner.run_on_video(
            frames_loader=frames_loader,
            detections_tracker=detections_tracker,
            depth_estimator=depth_estimator,
            draw_interval=args.draw_interval,
            save_path=save_path,
        )

        if save_path:
            results_file_path = save_path / "out_variables.pkl"
            with results_file_path.open("wb") as file:
                pickle.dump(slam_out, file)
                print(f"Saved the results to {results_file_path}")
        # Show results
        show_slam_out(slam_out=slam_out, save_path=args.save_path, example_path=example_path)


# ---------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
