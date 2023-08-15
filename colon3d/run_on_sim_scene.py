import argparse
import pickle
from pathlib import Path

import attrs
import h5py

from colon3d.show_slam_out import save_slam_out_plots
from colon3d.slam.alg_settings import AlgorithmParam
from colon3d.slam.slam_alg import SlamAlgRunner
from colon3d.util.data_util import SceneLoader
from colon3d.util.depth_egomotion import DepthAndEgoMotionLoader
from colon3d.util.general_util import ArgsHelpFormatter, Tee, bool_arg, create_empty_folder
from colon3d.util.performance_metrics import calc_performance_metrics, plot_trajectory_metrics
from colon3d.util.torch_util import to_default_type
from colon3d.util.tracks_util import DetectionsTracker

# ---------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(formatter_class=ArgsHelpFormatter)
    parser.add_argument(
        "--scene_path",
        type=str,
        default="data/sim_data/TestData21_cases/Scene_00000_0000",  # "data/sim_data/Zhang22/Scene_00000"
        help="Path to the scene folder",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="results/Temp/temp1",
        help="Path to the save outputs",
    )
    parser.add_argument(
        "--save_raw_outputs",
        type=bool_arg,
        default=False,
        help="If True then all the raw outputs will be saved (as pickle file), not just the plots and summary",
    )
    parser.add_argument(
        "--use_bundle_adjustment",
        type=bool_arg,
        default=True,
        help="If True then the bundle adjustment will be used to refine the trajectory",
    )
    parser.add_argument(
        "--depth_maps_source",
        type=str,
        default="none",
        choices=["ground_truth", "loaded_estimates", "online_estimates", "none"],
        help="The source of the depth maps, if 'ground_truth' then the ground truth depth maps will be loaded, "
        "if 'online_estimates' then the depth maps will be estimated online by the algorithm (using a pre-trained DepthNet)"
        "if 'loaded_estimates' then the depth maps estimations will be loaded, "
        "if 'none' then no depth maps will not be used,",
    )
    parser.add_argument(
        "--egomotions_source",
        type=str,
        default="none",
        choices=["ground_truth", "loaded_estimates", "online_estimates", "none"],
        help="The source of the egomotion, if 'ground_truth' then the ground truth egomotion will be loaded, "
        "if 'online_estimates' then the egomotion will be estimated online by the algorithm (using a pre-trained PoseNet)"
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
        "--draw_interval",
        type=int,
        default=50,
        help="plot and save figures each draw_interval frames",
    )
    parser.add_argument(
        "--save_overwrite",
        type=bool_arg,
        default=True,
        help="if True then the save folder will be overwritten",
    )
    args = parser.parse_args()
    print(f"args={args}")
    alg_settings_override = {} if args.use_bundle_adjustment else {"use_bundle_adjustment": False}

    slam_on_scene_runner = SlamOnSimSceneRunner(
        scene_path=Path(args.scene_path),
        save_path=Path(args.save_path),
        save_raw_outputs=args.save_raw_outputs,
        depth_maps_source=args.depth_maps_source,
        egomotions_source=args.egomotions_source,
        depth_and_egomotion_method=args.depth_and_egomotion_method,
        depth_and_egomotion_model_path=Path(args.depth_and_egomotion_model_path),
        alg_fov_ratio=args.alg_fov_ratio,
        n_frames_lim=args.n_frames_lim,
        alg_settings_override=alg_settings_override,
        draw_interval=args.draw_interval,
        save_overwrite=args.save_overwrite,
    )
    slam_on_scene_runner.run()


# ---------------------------------------------------------------------------------------------------------------------
@attrs.define
class SlamOnSimSceneRunner:
    scene_path: Path
    save_path: Path
    save_raw_outputs: bool
    depth_maps_source: str
    egomotions_source: str
    depth_and_egomotion_method: str
    depth_and_egomotion_model_path: Path
    alg_fov_ratio: float
    n_frames_lim: int
    alg_settings_override: dict | None = None
    draw_interval: int = 0
    save_overwrite: bool = True

    # ---------------------------------------------------------------------------------------------------------------------

    def run(self):
        self.scene_path = Path(self.scene_path)
        self.save_path = Path(self.save_path)
        is_created = create_empty_folder(self.save_path, save_overwrite=self.save_overwrite)
        if not is_created:
            print(f"{self.save_path} already exists...\n" + "-" * 50)
            return None
        print(f"Outputs will be saved to {self.save_path}")

        with Tee(self.save_path / "log_run_slam.txt"):  # save the prints to a file
            metrics_per_frame, metrics_stats = run_slam_on_scene(
                scene_path=self.scene_path,
                save_path=self.save_path,
                save_raw_outputs=self.save_raw_outputs,
                n_frames_lim=self.n_frames_lim,
                alg_fov_ratio=self.alg_fov_ratio,
                depth_maps_source=self.depth_maps_source,
                egomotions_source=self.egomotions_source,
                depth_and_egomotion_method=self.depth_and_egomotion_method,
                depth_and_egomotion_model_path=self.depth_and_egomotion_model_path,
                alg_settings_override=self.alg_settings_override,
                draw_interval=self.draw_interval,
                plot_names=None,  # create all plots
            )
        return metrics_per_frame, metrics_stats


# ---------------------------------------------------------------------------------------------------------------------


def run_slam_on_scene(
    scene_path: Path,
    save_path: Path,
    save_raw_outputs: bool,
    n_frames_lim: int,
    alg_fov_ratio: float,
    depth_maps_source: str,
    egomotions_source: str,
    depth_and_egomotion_method: str,
    depth_and_egomotion_model_path: str,
    alg_settings_override: dict | None = None,
    draw_interval: int = 0,
    plot_names: list | None = None,
):
    """ "
    Run the SLAM algorithm on a scene and save the results.
    Args:
        scene_path: path to the scene folder
        save_path: path to the save outputs
        n_frames_lim: upper limit on the number of frames used, if 0 then all frames are used
        alg_fov_ratio: If in range (0,1) then the algorithm will use only a fraction of the frames, if 0 then all of the frame is used.
        depth_maps_source: The source of the depth maps.
        egomotions_source: The source of the egomotion.
        draw_interval: plot and save figures each draw_interval frame, if 0 then no plots are saved.
        plot_names: list of plot names to save, if None then all plots are saved.

    """
    # get the default parameters for the SLAM algorithm
    alg_prm = AlgorithmParam()
    if alg_settings_override is not None:
        for k, v in alg_settings_override.items():
            setattr(alg_prm, k, v)

    scene_loader = SceneLoader(scene_path=scene_path, n_frames_lim=n_frames_lim, alg_fov_ratio=alg_fov_ratio)
    detections_tracker = DetectionsTracker(scene_path=scene_path, scene_loader=scene_loader)
    depth_estimator = DepthAndEgoMotionLoader(
        scene_path=scene_path,
        depth_maps_source=depth_maps_source,
        egomotions_source=egomotions_source,
        depth_and_egomotion_method=depth_and_egomotion_method,
        depth_and_egomotion_model_path=depth_and_egomotion_model_path,
        depth_lower_bound=alg_prm.depth_lower_bound,
        depth_upper_bound=alg_prm.depth_upper_bound,
        depth_default=alg_prm.depth_default,
    )

    # Run the SLAM algorithm
    slam_runner = SlamAlgRunner(
        alg_prm=alg_prm,
        scene_loader=scene_loader,
        detections_tracker=detections_tracker,
        depth_estimator=depth_estimator,
        save_path=save_path,
        draw_interval=draw_interval,
    )
    slam_out = slam_runner.run()

    if save_path and save_raw_outputs:
        results_file_path = save_path / "out_variables.pkl"
        # save results to a file
        with results_file_path.open("wb") as file:
            pickle.dump(slam_out, file)
            print(f"Saved the results to {results_file_path}")

    # create and save plots
    save_slam_out_plots(slam_out=slam_out, save_path=save_path, scene_path=scene_path, plot_names=plot_names)

    # load the  ground truth targets info
    targets_info_path = scene_path / "targets_info.pkl"
    if targets_info_path.exists():
        with targets_info_path.open("rb") as file:
            gt_targets_info = pickle.load(file)
    else:
        gt_targets_info = None
        print("No targets info file found...")

    # load the  ground-truth egomotions per frame (for evaluation)
    gt_data_path = scene_path / "gt_3d_data.h5"
    with h5py.File(gt_data_path.resolve(), "r") as hf:
        gt_cam_poses = to_default_type(hf["cam_poses"][:])  # load the ground-truth camera poses into memory

    # calculate performance metrics
    metrics_per_frame, metrics_stats = calc_performance_metrics(
        gt_cam_poses=gt_cam_poses,
        gt_targets_info=gt_targets_info,
        slam_out=slam_out,
    )
    metrics_stats["Example Name"] = scene_path.name
    plot_trajectory_metrics(metrics_per_frame=metrics_per_frame, save_path=save_path / "trajectory_metrics.png")

    print(f"Error metrics stats: {metrics_stats}")

    return metrics_per_frame, metrics_stats


# ---------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------------------------------------------------
