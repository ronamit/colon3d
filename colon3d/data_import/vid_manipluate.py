import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from colon3d.alg.tracks_loader import DetectionsTracker
from colon3d.util.data_util import SceneLoader
from colon3d.util.general_util import (
    ArgsHelpFormatter,
    create_empty_folder,
    save_rgb_image,
    save_video_from_frames_paths,
)

# ---------------------------------------------------------------------------------------------------------------------

# TODO: add the video rotation code from the colab notebook (including tracks)

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

    detections_tracker = DetectionsTracker(
        scene_path=load_scene_path,
        scene_loader=scene_loader,
    )
    # loop over all frames and saved them as images

    rgb_frames_paths = []
    fps = scene_loader.fps
    
    # save the frames
    frames_out_path = save_scene_path / "Frames"
    create_empty_folder(frames_out_path, save_overwrite=False)
    n_frames = scene_loader.n_frames
    
    n_tracks_per_frame = np.zeros(n_frames)
    for i_frame in range(n_frames):
        # Get the targets tracks in the current frame inside the algorithmic field of view
        tracks = detections_tracker.get_tracks_in_frame(i_frame, frame_type="alg_view")
        n_tracks_per_frame[i_frame] = len(tracks)
        
    
    
    # plot the number of tracks per frame
    plt.figure()
    plt.plot(n_tracks_per_frame)
    plt.xlabel("Frame index")
    plt.ylabel("Number of tracks")
    
    
    for i_frame in range(n_frames):
        print(f"i_frame={i_frame}/{n_frames}")
        im = scene_loader.get_frame_at_index(i_frame, frame_type="full")
        frame_name = f"{i_frame:06d}.png"
        # save the image
        save_rgb_image(img=im, save_path=frames_out_path / frame_name)
        rgb_frames_paths.append(frames_out_path / frame_name)

    # save the video
    save_video_from_frames_paths(save_path=save_scene_path / "Video", frames=rgb_frames_paths, fps=fps)



# ---------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------------------------------------------------
