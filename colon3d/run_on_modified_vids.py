from pathlib import Path

import numpy as np

from colon3d.modify_video import VideoModifier
from colon3d.run_on_scene import SlamRunner

# ---------------------------------------------------------------------------------------------------------------------


def main():
    base_data_path = Path("data") # Path("/mnt/disk1/data/my_videos")
    load_scene_path = base_data_path / "Example_4"
    alg_fov_ratio = 0.8

    video_modifier = VideoModifier(
        load_scene_path=load_scene_path,
        alg_fov_ratio=alg_fov_ratio,
        n_frames_lim=0,
        # verbose=False,
    )

    # set the grid of scales to modify the video with
    seg_scales =  np.array([2, 3, 4])
    
    for scale in seg_scales:
        # Modify the video to have longer out-of-view segments
        mod_scene_path = base_data_path / f"Mod_Vids/Scale_{scale}".replace(".", "_")
        video_modifier.run(seg_scale=scale, save_path=mod_scene_path)


        # Run the algorithm on the modified video
        slam_runner = SlamRunner(
            scene_path=mod_scene_path,
            save_path=mod_scene_path,
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
