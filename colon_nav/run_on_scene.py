import argparse
import pickle
from pathlib import Path

import attrs

from colon_nav.show_slam_out import save_slam_plots
from colon_nav.slam.alg_settings import AlgSettings
from colon_nav.slam.monocular_est_loader import DepthAndEgoMotionLoader
from colon_nav.slam.slam_alg import SlamAlgRunner
from colon_nav.slam.tracks_loader import DetectionsTracker
from colon_nav.util.data_util import SceneLoader
from colon_nav.util.general_util import ArgsHelpFormatter, Tee, bool_arg, create_empty_folder, to_path

# ---------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(formatter_class=ArgsHelpFormatter)
    parser.add_argument(
        "--scene_path",
        type=str,
        default="data/datasets/ColonNav/Test/Scene_00022/",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="data/results/temp_run_on_scene",
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
        "--model_path",
        type=str,
        default="data/models/EndoSFM_orig",
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
        "--print_interval",
        type=int,
        default=20,
        help="print the results each print_interval frames. If 0 then no prints",
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
        model_path=args.model_path,
        alg_fov_ratio=args.alg_fov_ratio,
        n_frames_lim=args.n_frames_lim,
        draw_interval=args.draw_interval,
        print_interval=args.print_interval,
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
    model_path: Path | None = None
    alg_fov_ratio: float = 0
    n_frames_lim: int = 0
    draw_interval: int = 0
    print_interval: int = 0

    # ---------------------------------------------------------------------------------------------------------------------
    def run(self):
        self.save_path = Path(self.save_path)
        self.scene_path = Path(self.scene_path)
        is_created = create_empty_folder(self.save_path, save_overwrite=self.save_overwrite)
        if not is_created:
            print(f"{self.save_path} already exists...\n" + "-" * 50)
            return None
        print(f"Outputs will be saved to {self.save_path}")

        assert self.scene_path.exists(), f"scene_path={self.scene_path} does not exist"

        with Tee(self.save_path / "log_run_slam.txt"):  # save the prints to a file
            # get the default parameters for the SLAM algorithm
            alg_prm = AlgSettings()

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
                scene_loader=scene_loader,
                depth_maps_source=self.depth_maps_source,
                egomotions_source=self.egomotions_source,
                depth_default=alg_prm.depth_default,
                model_path=to_path(self.model_path),
            )

            # Run the SLAM algorithm
            alg_runner = SlamAlgRunner(
                alg_prm=alg_prm,
                scene_loader=scene_loader,
                detections_tracker=detections_tracker,
                depth_and_ego_estimator=depth_estimator,
                save_path=self.save_path,
                draw_interval=self.draw_interval,
                print_interval=self.print_interval,
            )
            slam_out = alg_runner.run()

            if self.save_path and self.save_raw_outputs:
                results_file_path = self.save_path / "out_variables.pkl"
                with results_file_path.open("wb") as file:
                    pickle.dump(slam_out, file)
                    print(f"Saved the results to {results_file_path}")
            # Show results
            save_slam_plots(slam_out=slam_out, save_path=self.save_path, scene_path=self.scene_path)
        return slam_out


# ---------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------------------------------------------------
