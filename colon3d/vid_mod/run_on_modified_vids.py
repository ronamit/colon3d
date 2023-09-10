import argparse
from pathlib import Path

import numpy as np

from colon3d.run_on_scene import SlamRunner
from colon3d.util.general_util import ArgsHelpFormatter

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
    alg_fov_ratio = 0.8

    # set the grid of scales to modify the video with
    seg_scales = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    for scale in seg_scales:
        # Modify the video to have longer out-of-view segments
        mod_scene_path = base_save_path / f"Scale_{scale}".replace(".", "_")
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
        slam_runner.run()


# ---------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
# ---------------------------------------------------------------------------------------------------------------------
