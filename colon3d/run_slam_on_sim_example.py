import argparse
import pickle
from pathlib import Path

import h5py
import numpy as np

from colon3d.alg_settings import AlgorithmParam
from colon3d.data_util import FramesLoader
from colon3d.depth_util import DepthAndEgoMotionLoader
from colon3d.general_util import Tee, create_empty_folder
from colon3d.perfomance_metrics import calc_performance_metrics, plot_trajectory_metrics
from colon3d.show_slam_out import save_slam_out_plots
from colon3d.slam_alg import SlamRunner
from colon3d.tracks_util import DetectionsTracker

# ---------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--example_path",
        type=str,
        default="data/sim_data/SimData8_Examples/Seq_00002_0000",
        help=" path to the video",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="data/sim_data/SimData8_Examples/Seq_00002_0000/Results_Temp",
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
        metrics_per_frame, metrics_stats = run_slam_on_example(
            example_path=example_path,
            save_path=save_path,
            n_frames_lim=args.n_frames_lim,
            alg_fov_ratio=args.alg_fov_ratio,
            depth_maps_source=args.depth_maps_source,
            egomotions_source=args.egomotions_source,
            draw_interval=args.draw_interval,
            save_all_plots=True,
        )

# ---------------------------------------------------------------------------------------------------------------------


def run_slam_on_example(
    example_path: Path,
    save_path: Path,
    n_frames_lim: int,
    alg_fov_ratio: float,
    depth_maps_source: str,
    egomotions_source: str,
    draw_interval: int = 0,
    save_all_plots: bool = False,
    save_aided_nav_plot: bool = True,
):
    # get the default parameters for the SLAM algorithm
    alg_prm = AlgorithmParam()

    frames_loader = FramesLoader(sequence_path=example_path, n_frames_lim=n_frames_lim, alg_fov_ratio=alg_fov_ratio)
    detections_tracker = DetectionsTracker(example_path=example_path, frames_loader=frames_loader)
    depth_estimator = DepthAndEgoMotionLoader(
        example_path=example_path,
        depth_maps_source=depth_maps_source,
        egomotions_source=egomotions_source,
        alg_prm=alg_prm,
    )

    # Run the SLAM algorithm
    slam_runner = SlamRunner(alg_prm)
    slam_out = slam_runner.run(
        frames_loader=frames_loader,
        detections_tracker=detections_tracker,
        depth_estimator=depth_estimator,
        save_path=save_path,
        draw_interval=draw_interval,
    )

    if save_path:
        results_file_path = save_path / "out_variables.pkl"
        # save results to a file
        with results_file_path.open("wb") as file:
            pickle.dump(slam_out, file)
            print(f"Saved the results to {results_file_path}")

    if save_all_plots or save_aided_nav_plot:
        plot_names = None if save_all_plots else ["aided_nav"]
        save_slam_out_plots(slam_out=slam_out, save_path=save_path, example_path=example_path, plot_names=plot_names)

    # load the  ground truth targets info
    with (example_path / "targets_info.pkl").open("rb") as file:
        gt_targets_info = pickle.load(file)

    # load the  ground-truth egomotions per frame (for evaluation)
    with h5py.File(example_path / "gt_depth_and_egomotion.h5", "r") as hf:
        gt_cam_poses = np.array(hf["cam_poses"])

    # calculate performance metrics
    metrics_per_frame, metrics_stats = calc_performance_metrics(
        gt_cam_poses=gt_cam_poses,
        gt_targets_info=gt_targets_info,
        slam_out=slam_out,
    )
    metrics_stats["Example Name"] = example_path.name
    plot_trajectory_metrics(metrics_per_frame=metrics_per_frame, save_path=save_path / "trajectory_metrics.png")

    print(f"Error metrics stats: {metrics_stats}")

    return metrics_per_frame, metrics_stats


# ---------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
