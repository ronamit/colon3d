import argparse
import pickle
from pathlib import Path

import h5py

from colon3d.run_on_scene import SlamRunner
from colon3d.util.general_util import ArgsHelpFormatter
from colon3d.util.performance_metrics import calc_performance_metrics, plot_trajectory_metrics
from colon3d.util.data_util import SceneLoader, get_origin_scene_path

# ---------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(formatter_class=ArgsHelpFormatter)
    parser.add_argument(
        "--base_save_path",
        type=str,
        default="/mnt/disk1/results/Mod_Vids",
        help="path to the save outputs",
    )
    args = parser.parse_args()
    base_save_path = Path(args.base_save_path)

    #  get the per-scale scene folders paths
    mod_scene_paths = sorted(base_save_path.glob("Scale_*"))

    for mod_scene_path in mod_scene_paths:
        # get info about the scene:
        with (mod_scene_path / "out_of_view_info.pkl").open("rb") as f:
            info_dict = pickle.load(f)
            scale = info_dict["scale"]
            new_segments = info_dict["new_segments"]
            is_in_view_new = info_dict["is_in_view_new"]
            alg_fov_ratio = info_dict["alg_fov_ratio"] if "alg_fov_ratio" in info_dict else 0.8
        print(
            "-" * 100
            + f"\nscale={scale}, in_view_frames={is_in_view_new.sum()}, out_of_view_frames={len(is_in_view_new) - is_in_view_new.sum()}",
        )

        # Modify the video to have longer out-of-view segments
        mod_scene_results_path = mod_scene_path / "results"

        # Run the algorithm on the modified video
        slam_runner = SlamRunner(
            scene_path=mod_scene_path,
            save_path=mod_scene_results_path,
            save_overwrite=True,
            save_raw_outputs=False,
            depth_maps_source="none",
            egomotions_source="none",
            depth_and_egomotion_method="none",
            alg_fov_ratio=alg_fov_ratio,
            n_frames_lim=0,
            draw_interval=200,
            verbose_print_interval=0,
        )
        slam_out = slam_runner.run()
        online_est_track_world_loc = slam_out["online_est_track_world_loc"]
        online_est_track_cam_loc = slam_out["online_est_track_cam_loc"]
        online_est_track_angle = slam_out["online_est_track_angle"]
        
        # save the estimated track location in the camera system per frame
        with pickle.open(mod_scene_results_path / "slam_results.pkl", "wb") as f:
            save_dict = {"online_est_track_world_loc": online_est_track_world_loc,
                         "online_est_track_cam_loc": online_est_track_cam_loc,
                         "online_est_track_angle": online_est_track_angle}
            pickle.dump(save_dict, f)

 

# ---------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
# ---------------------------------------------------------------------------------------------------------------------
