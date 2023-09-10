import argparse
import shutil
from pathlib import Path

import attrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from colon3d.alg.tracks_loader import DetectionsTracker
from colon3d.util.data_util import SceneLoader
from colon3d.util.general_util import (
    ArgsHelpFormatter,
    create_empty_folder,
    save_plot_and_close,
    save_video_from_frames_list,
)
from colon3d.visuals.plots_2d import draw_alg_view_in_the_full_frame, draw_track_box_on_frame


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
        "--save_path",
        type=str,
        default="/mnt/disk1/data/my_videos/Example_4_modified",
        help="Path to save modified video",
    )
    parser.add_argument(
        "--seg_scale",
        type=float,
        default=2.5,
        help="Scaling factor to increase the time of the out-of-view segments (must be >- 1)",
    )

    args = parser.parse_args()
    video_modifier = VideoModifier(load_scene_path=Path(args.load_scene_path))
    video_modifier.run(save_path=Path(args.save_path), seg_scale=args.seg_scale)


# ---------------------------------------------------------------------------------------------------------------------


@attrs.define
class VideoModifier:
    load_scene_path: Path = Path()
    # the ratio of the algorithmic field of view (alg_view) to the full field of view (full_view):
    alg_fov_ratio: float = 0.8
    n_frames_lim: int = 0  # limit the number of frames to load from the original video
    max_len = 5000  # limit the number of frames in the new video
    verbose: bool = True
    # Attributes that are set in __attrs_post_init__:
    scene_loader: SceneLoader = attrs.field(init=False)
    detections_tracker: DetectionsTracker = attrs.field(init=False)
    alg_view_cropper: np.ndarray = attrs.field(init=False)
    n_frames: int = attrs.field(init=False)
    is_in_view: np.ndarray = attrs.field(init=False)
    segments: list = attrs.field(init=False)
    first_frame_to_use: int = attrs.field(init=False)

    # ---------------------------------------------------------------------------------------------------------------------
    def __attrs_post_init__(self):
        self.scene_loader = SceneLoader(
            scene_path=self.load_scene_path,
            n_frames_lim=self.n_frames_lim,
            alg_fov_ratio=self.alg_fov_ratio,
        )

        self.detections_tracker = DetectionsTracker(
            scene_path=self.load_scene_path,
            scene_loader=self.scene_loader,
        )
        self.alg_view_cropper = self.scene_loader.alg_view_cropper

        self.n_frames = self.scene_loader.n_frames
        # if track is in view, then 1, else 0
        self.is_in_view = np.zeros(self.n_frames, dtype=bool)

        # save the start and end frame indexes of each out-of-view segment as list
        self.segments = []

        for i_frame in range(self.n_frames):
            # Get the targets tracks in the current frame inside the algorithmic field of view
            tracks = self.detections_tracker.get_tracks_in_frame(i_frame, frame_type="alg_view")
            self.is_in_view[i_frame] = len(tracks) > 0
            if self.is_in_view[i_frame]:
                # if current frame is in view and previous frame was out of view, save the end of the out-of-view segment
                if i_frame > 0 and not self.is_in_view[i_frame - 1]:
                    self.segments[-1]["last"] = i_frame - 1
            # if current frame is out of view and it is frame 0 or if the previous frame was in view, save the start of the out-of-view segment
            elif i_frame == 0 or (i_frame > 0 and self.is_in_view[i_frame - 1]):
                self.segments.append({"first": i_frame, "last": None})
            # if we reached the last frame and it is not in view, save the end of the out-of-view segment
            if i_frame == self.n_frames - 1 and not self.is_in_view[i_frame]:
                self.segments[-1]["last"] = i_frame

        # start the new video with the first frame that is in view in the original video
        self.first_frame_to_use = self.is_in_view.argmax()

        # Average segment duration
        for seg in self.segments:
            seg["duration"] = 1 + seg["last"] - seg["first"]
        avg_segment_duration = np.mean(np.array([seg["duration"] for seg in self.segments]))
        print(f"Average segment duration: {avg_segment_duration:.2f} frames")

    # ---------------------------------------------------------------------------------------------------------------------
    def run(self, seg_scale: float, save_path: Path):
        """"""
        """ Modify the video to have longer out-of-view segments
        Args:
            seg_scale: scale factor to change the duration of each out-of-view segments
        """
        assert seg_scale >= 1, f"seg_scale={seg_scale} must be >= 1"

        create_empty_folder(save_path, save_overwrite=True)
        if self.verbose:
            print("Out of view segments:")
            for i_seg, segment in enumerate(self.segments):
                print(f"Out of view segment #{i_seg}: {segment}")

            plt.figure()
            plt.plot(self.is_in_view)
            plt.xlabel("Frame index")
            plt.ylabel("Is track in alg. view")
            save_plot_and_close(save_path / "orig_is_in_view.png")

        # # Extend the time of each out-of-view segment, by playing it forward and backward
        new_vid_frame_inds = []
        # number of frames that remains to be added from the current out-of-view segment to the new video:
        n_frames_remains = -1  # -1 means that we are not in an out-of-view segment
        move_dir = 1  # 1 for forward, -1 for backward
        i = 0
        i_frame = self.first_frame_to_use
        forward_only = False
        cur_seg_idx = -1

        # go over all the original video frames and add them to the new video in a way that out-of-view segments may be repeated
        while (i_frame < self.n_frames) and (i < self.max_len):
            if self.is_in_view[i_frame]:
                move_dir = 1  # move forward
            else:  # We are in an out-of-view segment.
                i_seg, seg = self.get_orig_vid_out_of_view_segment(i_frame)
                if i_seg > cur_seg_idx:
                    # we started a new out-of-view segment
                    print(f"** i_frame={i_frame}, staring a new out-of-view segment: #{i_seg}, {seg}")
                    n_frames_remains = int(seg["duration"] * seg_scale) - 1
                    print(f"n_frames_remains={n_frames_remains}")
                    move_dir = 1  # move forward
                    forward_only = False
                    cur_seg_idx = i_seg

                elif i_frame in {seg["first"], seg["last"]}:
                    print(f"Reached on the ends of the current out-of-view segment #{i_seg}, i_frame={i_frame}")
                    if n_frames_remains > 0:
                        print("Change direction of movement")
                        move_dir *= -1  # change the direction of movement

                elif n_frames_remains <=  (seg["last"] - i_frame + 1):
                    #  we may still have frames to add for this segment, but we must stop going back and forth and go forward only
                    # so that we wil reach the end of the segment when n_frames_remains==0
                    # i.e. we don't have enough frames to go back and forth and still reach the end of the segment
                    move_dir = 1  # move forward
                    forward_only = True
                    
                else:
                    pass  # keep the current direction of movement

                # in any case - decrease the number of frames that remains to be added from the current out-of-view segment to the new video
                n_frames_remains -= 1
                print(f"--i={i}, original i_frame={i_frame}, move_dir={move_dir}, n_frames_remains={n_frames_remains}, forward_only={forward_only}")

            # add the current frame to the new video
            new_vid_frame_inds.append(i_frame)
            i_frame += move_dir
            i += 1

        print(f"Number of frames in the original video: {self.n_frames}")

        n_frames_new = len(new_vid_frame_inds)
        print(f"Number of frames in the new video: {n_frames_new}")

        # plot new_vid_frame_inds:
        plt.figure()
        plt.plot(new_vid_frame_inds)
        plt.xlabel("Frame index in the new video")
        plt.ylabel("Frame index in the original video")
        save_plot_and_close(save_path / "new_vid_frame_inds.png")

        # Create new tracks file
        new_tracks_original = {"frame_idx": [], "track_id": [], "xmin": [], "ymin": [], "xmax": [], "ymax": []}

        # Create new video
        fps = self.scene_loader.fps

        # save the frames
        frames_out_path = save_path / "Frames"
        create_empty_folder(frames_out_path, save_overwrite=True)

        frames = []
        vis_frames = []

        print("Creating new video...")
        for i in range(n_frames_new):
            print(f"i_frame={i}/{n_frames_new}", end="\r", flush=True)
            i_frame = new_vid_frame_inds[i]  # index of the current frame in the original video

            # save the frame to the new video frame list
            frame = self.scene_loader.get_frame_at_index(i_frame, frame_type="full")
            frames.append(frame)

            #  save the tracks info to the new tracks file
            orig_tracks = self.detections_tracker.get_tracks_in_frame(i_frame, frame_type="full_view")
            for cur_track in orig_tracks.values():
                for k, v in cur_track.items():
                    new_tracks_original[k].append(v)
                new_tracks_original["frame_idx"][-1] = i  # change the frame index to the new frame index

            ##  Create the visualization of the algorithm view + bounding boxes for the new video
            vis_frame = np.copy(frame)
            # draw the algorithm view circle
            vis_frame = draw_alg_view_in_the_full_frame(vis_frame, self.scene_loader)
            # draw bounding boxes for the tracks in the algorithm view (in the full frame)
            alg_view_tracks = self.detections_tracker.get_tracks_in_frame(i_frame, frame_type="alg_view")
            for track_id, cur_track in alg_view_tracks.items():
                vis_frame = draw_track_box_on_frame(
                    rgb_frame=vis_frame,
                    track=cur_track,
                    track_id=track_id,
                    alg_view_cropper=self.alg_view_cropper,
                    convert_from_alg_view_to_full=True,
                )
            vis_frames.append(vis_frame)

        # save the videos
        save_video_from_frames_list(save_path=save_path / "Video", frames=frames, fps=fps)
        save_video_from_frames_list(save_path=save_path / "Video_Vis", frames=vis_frames, fps=fps)

        # Copy the rest of the files from the original scene folder (those are unchanged by the modification)
        files_to_copy = ["meta_data.yaml"]
        for file_name in files_to_copy:
            shutil.copy(self.load_scene_path / file_name, save_path / file_name)

        # save the new tracks csv file
        pd.DataFrame(new_tracks_original).to_csv(save_path / "tracks.csv", index=False)

    # ---------------------------------------------------------------------------------------------------------------------
    def get_orig_vid_out_of_view_segment(self, i_frame: int) -> dict:
        # find the segment:
        for i_seg, seg in enumerate(self.segments):
            if seg["first"] <= i_frame <= seg["last"]:
                return i_seg, seg
        return None, None


# ---------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------------------------------------------------
