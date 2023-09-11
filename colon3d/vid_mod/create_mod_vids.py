import argparse
import pickle
from pathlib import Path

import numpy as np

from colon3d.util.general_util import ArgsHelpFormatter, create_empty_folder
from colon3d.vid_mod.modify_video import VideoModifier

# ---------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(formatter_class=ArgsHelpFormatter)
    parser.add_argument(
        "--load_scene_path",
        type=str,
        default="/mnt/disk1/data/my_videos/Example_4",
        help="Path to load scene",
    )
    parser.add_argument(
        "--base_save_path",
        type=str,
        default="/mnt/disk1/results/Mod_Vids",
        help="path to the save outputs",
    )
    args = parser.parse_args()
    load_scene_path = Path(args.load_scene_path)
    base_save_path = Path(args.base_save_path)
    alg_fov_ratio = 0.8

    video_modifier = VideoModifier(
        load_scene_path=load_scene_path,
        alg_fov_ratio=alg_fov_ratio,
        n_frames_lim=0,
        # verbose=False,
    )

    # set the grid of scales to modify the video with
    seg_scales = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    for scale in seg_scales:
        # Modify the video to have longer out-of-view segments
        mod_scene_path = base_save_path / f"Scale_{scale}".replace(".", "_")
        
        # create the modified scene folder
        create_empty_folder(mod_scene_path, save_overwrite=True)
        
        # create the modified video
        video_modifier.run(seg_scale=scale, save_path=mod_scene_path)

        # save the scale info as a pickle file
        with (mod_scene_path / "out_of_view_info.pkl").open("wb") as f:
            info_dict = {"scale": scale,  "new_segments": video_modifier.new_segments, "is_in_view_new": video_modifier.is_in_view_new}
            pickle.dump(info_dict, f)
        
        with (mod_scene_path / "out_of_view_info.txt").open("w") as f:
            f.write(f"scale: {scale}\n")
            f.write(f"n_frames_new: {video_modifier.n_frames_new}\n")
            f.write(f"n_frames_out_of_view: {video_modifier.is_in_view_new.sum()}\n")
            f.write(f"new_segments: {video_modifier.new_segments}\n")
# ---------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
# ---------------------------------------------------------------------------------------------------------------------
