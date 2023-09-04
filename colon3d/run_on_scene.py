import argparse
import pickle
from pathlib import Path

import attrs

from colon3d.alg.alg_settings import AlgorithmParam
from colon3d.alg.monocular_est_loader import DepthAndEgoMotionLoader
from colon3d.alg.slam_alg import SlamAlgRunner
from colon3d.alg.tracks_loader import DetectionsTracker
from colon3d.show_slam_out import save_slam_plots
from colon3d.util.data_util import SceneLoader
from colon3d.util.general_util import ArgsHelpFormatter, Tee, bool_arg, create_empty_folder, to_path

# ---------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(formatter_class=ArgsHelpFormatter)
    parser.add_argument(
        "--scene_path",
        type=str,
        default="/mnt/disk1/data/my_videos/Example_4_yoyo",  # Example_4_rotV2 | Example_4 | Example_4_yoyo
        help="path to the scene folder",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/mnt/disk1/results/my_videos_results/Example_4_new",
        help="path to the save outputs",
    )
    parser.add_argument(
        "--save_overwrite",
        type=bool_arg,
        default=True,
        help="If True then the save folder will be overwritten if exists",
    )
    parser.add_argument(
        "--save_raw_outputs",
        type=bool_arg,
        default=False,
        help="If True then all the raw outputs will be saved (as pickle file), not just the plots",
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
        "--depth_and_egomotion_method",
        type=str,
        default="EndoSFM",
        choices=["EndoSFM", "MonoDepth2"],
        help="The method used for depth and egomotion estimation (to be used for the case of online estimation))",
    )
    parser.add_argument(
        "--depth_and_egomotion_model_path",
        type=str,
        default="/mnt/disk1/saved_models/EndoSFM_orig",
        help="path to the saved depth and egomotion model (PoseNet and DepthNet) to be used for the case of online estimation",
    )
    parser.add_argument(
        "--alg_fov_ratio",
        type=float,
        default=0.8,
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
        default=200,
        help="plot and save figures each draw_interval frames",
    )
    parser.add_argument(
        "--verbose_print_interval",
        type=int,
        default=0,
        help="print verbose information each verbose_print_interval frames",
    )

    args = parser.parse_args()
    print(f"args={args}")
    slam_runner = SlamRunner(
        scene_path=args.scene_path,
        save_path=args.save_path,
        save_overwrite=args.save_overwrite,
        save_raw_outputs=args.save_raw_outputs,
        depth_maps_source=args.depth_maps_source,
        egomotions_source=args.egomotions_source,
        depth_and_egomotion_method=args.depth_and_egomotion_method,
        depth_and_egomotion_model_path=args.depth_and_egomotion_model_path,
        alg_fov_ratio=args.alg_fov_ratio,
        n_frames_lim=args.n_frames_lim,
        draw_interval=args.draw_interval,
        verbose_print_interval=args.verbose_print_interval,
    )
    slam_runner.run()


# ---------------------------------------------------------------------------------------------------------------------


@attrs.define
class SlamRunner:
    scene_path: Path
    save_path: Path
    save_overwrite: bool
    save_raw_outputs: bool
    depth_maps_source: str
    egomotions_source: str
    depth_and_egomotion_method: str | None = None
    depth_and_egomotion_model_path: Path | None = None
    alg_fov_ratio: float = 0
    n_frames_lim: int = 0
    draw_interval: int = 0
    verbose_print_interval: int = 0

    # ---------------------------------------------------------------------------------------------------------------------
    def run(self):
        self.save_path = Path(self.save_path)
        self.scene_path = Path(self.scene_path)
        is_created = create_empty_folder(self.save_path, save_overwrite=self.save_overwrite)
        if not is_created:
            print(f"{self.save_path} already exists...\n" + "-" * 50)
            return
        print(f"Outputs will be saved to {self.save_path}")

        assert self.scene_path.exists(), f"scene_path={self.scene_path} does not exist"

        with Tee(self.save_path / "log_run_slam.txt"):  # save the prints to a file
            # get the default parameters for the SLAM algorithm
            alg_prm = AlgorithmParam()

            scene_loader = SceneLoader(
                scene_path=self.scene_path,
                n_frames_lim=self.n_frames_lim,
                alg_fov_ratio=self.alg_fov_ratio,
            )
            detections_tracker = DetectionsTracker(
                scene_path=self.scene_path,
                scene_loader=scene_loader,
            )
            depth_estimator = DepthAndEgoMotionLoader(
                scene_path=self.scene_path,
                depth_maps_source=self.depth_maps_source,
                egomotions_source=self.egomotions_source,
                depth_and_egomotion_method=self.depth_and_egomotion_method,
                depth_and_egomotion_model_path=to_path(self.depth_and_egomotion_model_path),
                depth_lower_bound=alg_prm.depth_lower_bound,
                depth_upper_bound=alg_prm.depth_upper_bound,
                depth_default=alg_prm.depth_default,
            )

            # Run the SLAM algorithm
            alg_runner = SlamAlgRunner(
                alg_prm=alg_prm,
                scene_loader=scene_loader,
                detections_tracker=detections_tracker,
                depth_and_ego_estimator=depth_estimator,
                save_path=self.save_path,
                draw_interval=self.draw_interval,
                verbose_print_interval=self.verbose_print_interval,
            )
            slam_out = alg_runner.run()

            if self.save_path and self.save_raw_outputs:
                results_file_path = self.save_path / "out_variables.pkl"
                with results_file_path.open("wb") as file:
                    pickle.dump(slam_out, file)
                    print(f"Saved the results to {results_file_path}")
            # Show results
            save_slam_plots(slam_out=slam_out, save_path=self.save_path, scene_path=self.scene_path)


# ---------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------------------------------------------------
