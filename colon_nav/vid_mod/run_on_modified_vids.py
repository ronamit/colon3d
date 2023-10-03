import argparse
import pickle
from pathlib import Path

from colon_nav.run_on_scene import SlamRunner
from colon_nav.util.general_util import ArgsHelpFormatter

# ---------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(formatter_class=ArgsHelpFormatter)
    parser.add_argument(
        "--base_load_path",
        type=str,
        default="data/datasets/real_videos/Mod_Vids",
        help="path to the save outputs",
    )
    parser.add_argument(
        "--base_save_path",
        type=str,
        default="data/results/real_videos/Mod_Vids",
        help="path to the save outputs",
    )
    args = parser.parse_args()
    base_load_path = Path(args.base_load_path)
    base_save_path = Path(args.base_save_path)

    #  get the per-scale scene folders paths
    mod_scene_paths = sorted(base_load_path.glob("Scale_*"))

    for mod_scene_path in mod_scene_paths:
        # get info about the scene:
        with (mod_scene_path / "scene_info.pkl").open("rb") as f:
            scene_info = pickle.load(f)
            seg_scale = scene_info["seg_scale"]
            # new_segments = info_dict["new_segments"]
            is_in_view_new = scene_info["is_in_view_new"]
            alg_fov_ratio = scene_info["alg_fov_ratio"] if "alg_fov_ratio" in scene_info else 0.8
        print(
            "-" * 100
            + f"\nscale={seg_scale}, in_view_frames={is_in_view_new.sum()}, out_of_view_frames={len(is_in_view_new) - is_in_view_new.sum()}",
        )

        # Modify the video to have longer out-of-view segments
        mod_scene_results_path = base_save_path / f"Scale_{seg_scale}".replace(".", "_")

        # Run the algorithm on the modified video
        slam_runner = SlamRunner(
            scene_path=mod_scene_path,
            save_path=mod_scene_results_path,
            save_overwrite=True,
            save_raw_outputs=False,
            depth_maps_source="none",
            egomotions_source="none",
            model_name="none",
            alg_fov_ratio=alg_fov_ratio,
            n_frames_lim=0,
            draw_interval=200,
            print_interval=0,
        )
        slam_out = slam_runner.run()
        online_est_track_world_loc = slam_out["online_est_track_world_loc"]
        online_est_track_cam_loc = slam_out["online_est_track_cam_loc"]
        online_est_track_angle = slam_out["online_est_track_angle"]

        # save the estimated track location in the camera system per frame

        with (mod_scene_results_path / "slam_results.pkl").open("wb") as f:
            save_dict = {
                "online_est_track_world_loc": online_est_track_world_loc,
                "online_est_track_cam_loc": online_est_track_cam_loc,
                "online_est_track_angle": online_est_track_angle,
                "scene_info": scene_info,
            }
            pickle.dump(save_dict, f)


# ---------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
# ---------------------------------------------------------------------------------------------------------------------
