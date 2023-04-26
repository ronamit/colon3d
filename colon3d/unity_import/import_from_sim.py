import argparse
from pathlib import Path

from colon3d.general_util import create_empty_folder
from colon3d.unity_import.sim_import_util import (
    create_metadata,
    save_ground_truth_depth_and_cam_poses,
    save_rgb_video,
)

# --------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_sim_data_path",
        type=str,
        default="data/raw_sim_data/Seq_00009",
        help="The path to the Unity simulator generated data of a single sequence",
    )
    parser.add_argument(
        "--path_to_save_sequence",
        type=str,
        default="data/sim_data/Seq_00009_short",
        help="The path to save the processed simulated sequence",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=0,
        help="frame rate in Hz of the output videos, if 0 the frame rate will be extracted from the settings file",
    )
    parser.add_argument(
        "--limit_n_frames",
        type=int,
        default=200,
        help="The number of frames to process, if 0 all frames will be processed",
    )
    args = parser.parse_args()
    seq_in_path = Path(args.raw_sim_data_path)
    seq_out_path = Path(args.path_to_save_sequence)
    print("Raw simulated sequences will be be loaded from: ", seq_in_path)
    create_empty_folder(seq_out_path, ask_overwrite=True)
    print(f"The processed sequence will be saved to {seq_out_path}")
    metadata, n_frames = create_metadata(seq_in_path, seq_out_path, args)
    fps = metadata["fps"]
    if args.limit_n_frames > 0:
        n_frames = min(args.limit_n_frames, n_frames)
        print(f"Only {n_frames} frames will be processed")
    else:
        print(f"All {n_frames} frames will be processed")
    save_ground_truth_depth_and_cam_poses(
        seq_in_path=seq_in_path,
        seq_out_path=seq_out_path,
        metadata=metadata,
        n_frames=n_frames,
    )

    save_rgb_video(
        seq_in_path=seq_in_path,
        seq_out_path=seq_out_path,
        vid_file_name="Video",
        fps=fps,
        n_frames=n_frames,
    )


# --------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
