import argparse
from pathlib import Path

from colon3d.util.data_util import SceneLoader
from colon3d.util.general_util import (
    ArgsHelpFormatter,
    create_empty_folder,
    save_rgb_image,
    save_video_from_frames_paths,
)

# ---------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(formatter_class=ArgsHelpFormatter)
    parser.add_argument(
        "--load_scene_path",
        type=str,
        default="/mnt/disk1/data/my_videos/Example_4",  # "data/my_videos/Example_4_rotV2",
        help="Path to the scene folder to load",
    )
    parser.add_argument(
        "--save_scene_path",
        type=str,
        default="/mnt/disk1/data/my_videos/Example_4_M",
        help=" Path to the scene folder to save",
    )
    args = parser.parse_args()
    print(f"args={args}")

    load_scene_path = Path(args.load_scene_path)
    save_scene_path = Path(args.save_scene_path)

    alg_fov_ratio = 0.8
    n_frames_lim = 0
    scene_loader = SceneLoader(
        scene_path=load_scene_path,
        n_frames_lim=n_frames_lim,
        alg_fov_ratio=alg_fov_ratio,
    )

    # loop over all frames and saved them as images

    rgb_frames_paths = []
    fps = 0.1

    # save the frames
    frames_out_path = save_scene_path / "Frames"
    create_empty_folder(frames_out_path, save_overwrite=False)
    n_frames = len(rgb_frames_paths)
    for i_frame in range(n_frames):
        im = scene_loader.get_frame_at_index(i_frame, frame_type="full")
        frame_name = f"{i_frame:06d}.png"
        # save the image
        save_rgb_image(img=im, save_path=frames_out_path / frame_name)

    # save the video
    save_video_from_frames_paths(save_path=save_scene_path / "Video", frames=rgb_frames_paths, fps=fps)


# ----------------------------------- gb----------------------------------------------------------------------------------
 
if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------------------------------------------------
