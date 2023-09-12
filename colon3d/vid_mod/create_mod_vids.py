import argparse
from pathlib import Path

import numpy as np

from colon3d.util.general_util import ArgsHelpFormatter, create_empty_folder
from colon3d.vid_mod.modify_video import VideoModifier

# ---------------------------------------------------------------------------------------------------------------------

""
"Create modified videos with different times scales of the out-of-view segments by augmenting the original scene"
""


def main():
    parser = argparse.ArgumentParser(formatter_class=ArgsHelpFormatter)
    parser.add_argument(
        "--load_scene_path",
        type=str,
        default="data_gcp/datasets/real_videos/Example_4",
        help="Path to load scene",
    )
    parser.add_argument(
        "--base_save_path",
        type=str,
        default="data/datasets/real_videos/Mod_Vids",
        help="path to the save outputs",
    )
    parser.add_argument(
        "--alg_fov_ratio",
        type=float,
        default=0.8,
        help="The ratio of the field of view of the algorithm",
    )
    args = parser.parse_args()
    load_scene_path = Path(args.load_scene_path)
    base_save_path = Path(args.base_save_path)

    video_modifier = VideoModifier(
        load_scene_path=load_scene_path,
        alg_fov_ratio=args.alg_fov_ratio,
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


# ---------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
# ---------------------------------------------------------------------------------------------------------------------
