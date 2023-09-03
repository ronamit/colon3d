from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from colon3d.alg.tracks_loader import DetectionsTracker
from colon3d.util.data_util import SceneLoader
from colon3d.util.general_util import create_empty_folder, save_plot_and_close, save_video_from_frames_list
from colon3d.visuals.plots_2d import draw_alg_view_in_the_full_frame, draw_track_box_on_frame

load_scene_path = Path("/mnt/disk1/data/my_videos/Example_4")
save_scene_path = Path("/mnt/disk1/data/my_videos/Example_4_processed")
alg_fov_ratio = 0.8
n_frames_lim = 0  # limit the number of frames to load from the original video

seg_scale = 3  # scale factor to change the duration of each out-of-view segments
max_len = 5000  # limit the number of frames in the new video

scene_loader = SceneLoader(
    scene_path=load_scene_path,
    n_frames_lim=n_frames_lim,
    alg_fov_ratio=alg_fov_ratio,
)

detections_tracker = DetectionsTracker(
    scene_path=load_scene_path,
    scene_loader=scene_loader,
)
alg_view_cropper = scene_loader.alg_view_cropper


n_frames = scene_loader.n_frames
# if track is in view, then 1, else 0
is_in_view = np.zeros(n_frames, dtype=bool)

# save the start and end frame indexes of each out-of-view segment as list
segments = []

for i_frame in range(n_frames):
    # Get the targets tracks in the current frame inside the algorithmic field of view
    tracks = detections_tracker.get_tracks_in_frame(i_frame, frame_type="alg_view")
    is_in_view[i_frame] = len(tracks) > 0
    if is_in_view[i_frame]:
        # if current frame is in view and previous frame was out of view, save the end of the out-of-view segment
        if i_frame > 0 and not is_in_view[i_frame - 1]:
            segments[-1]["last"] = i_frame - 1
    # if current frame is out of view and it is frame 0 or if the previous frame was in view, save the start of the out-of-view segment
    elif i_frame == 0 or (i_frame > 0 and is_in_view[i_frame - 1]):
        segments.append({"first": i_frame, "last": None})
    # if we reached the last frame and it is not in view, save the end of the out-of-view segment
    if i_frame == n_frames - 1 and not is_in_view[i_frame]:
        segments[-1]["last"] = i_frame


# plot the number of tracks per frame
plt.figure()
plt.plot(is_in_view)
plt.xlabel("Frame index")
plt.ylabel("Is track in alg. view")
save_plot_and_close(save_scene_path / "orig_is_in_view.png")


# start the new video with the first frame that is in view in the original video
first_frame_to_use = is_in_view.argmax()

# save the start and end frame indexes of each out-of-view segment as list
segments = []

for i_frame in range(first_frame_to_use, n_frames):
    if is_in_view[i_frame]:
        # if current frame is in view and previous frame was out of view, save the end of the out-of-view segment
        if i_frame > 0 and not is_in_view[i_frame - 1] and len(segments) > 0:
            segments[-1]["last"] = i_frame - 1
    # if current frame is out of view and it is frame 0 or if the previous frame was in view, save the start of the out-of-view segment
    elif i_frame == 0 or (i_frame > 0 and is_in_view[i_frame - 1]):
        segments.append({"first": i_frame, "last": None})
    # if we reached the last frame and it is not in view, save the end of the out-of-view segment
    if i_frame == n_frames - 1 and not is_in_view[i_frame]:
        segments[-1]["last"] = i_frame


print("Out of view segments:")
for i_seg, segment in enumerate(segments):
    print(f"Out of view segment #{i_seg}: {segment}")


# Average segment duration
for seg in segments:
    seg["duration"] = 1 + seg["last"] - seg["first"]
avg_segment_duration = np.mean(np.array([seg["duration"] for seg in segments]))
print(f"Average segment duration: {avg_segment_duration:.2f} frames")

# # Extend the time of each out-of-view segment, by playing it forward and backward


new_vid_frame_inds = []


next_seg_idx = 0 # index of the out-of-view segment we will see next
seg = segments[next_seg_idx]
n_frames_remains = 0  # number of frames that remains to be added from the current out-of-view segment to the new video
move_dir = 1  # 1 for forward, -1 for backward
i = 0
i_frame = first_frame_to_use

# go over all the original video frames and add them to the new video in a way that out-of-view segments may be repeated
while (i_frame < n_frames) and (i < max_len):
    print(f"i_frame={i_frame}/{n_frames}, n_frames_remains={n_frames_remains}, segment #{next_seg_idx}: {seg}, is_in_view={is_in_view[i_frame]}, move_dir={move_dir}")
    if is_in_view[i_frame]:
        move_dir = 1  # move forward
    else: # We are in an out-of-view segment.
        print(f"Position in the original out-of-view segment: {i_frame - seg['first'] + 1}/{seg['duration']}")
        if n_frames_remains == 0:
            # we started a new out-of-view segment
            seg = segments[next_seg_idx]
            next_seg_idx += 1
            move_dir = 1  # move forward
            n_frames_remains = int(seg_scale * seg["duration"])
        elif n_frames_remains > seg["duration"] // 2:
            # if we still have frames to add for this segment, and to do back and forth
            if i_frame in {seg["first"] , seg["last"]}:
                print(f"Reached on the ends of the current out-of-view segment #{next_seg_idx}, so we will change the direction of movement")
                move_dir *= -1
        else: #  0 < n_frames_remains <= seg["duration"] // 2:
            #  if we still have frames to add for this segment, but we must stop going back and forth and go forward only
            # so that we wil reach the end of the segment when n_frames_remains==0
            move_dir = 1  # move forward
        n_frames_remains -= 1
    # add the current frame to the new video
    new_vid_frame_inds.append(i_frame)
    i_frame += move_dir
    i += 1


print(f"Number of frames in the original video: {n_frames}")

n_frames_new = len(new_vid_frame_inds)
print(f"Number of frames in the new video: {n_frames_new}")


# plot new_vid_frame_inds:
plt.figure()
plt.plot(new_vid_frame_inds)
plt.xlabel("Frame index in the new video")
plt.ylabel("Frame index in the original video")
save_plot_and_close(save_scene_path / "new_vid_frame_inds.png")



# # Create new video

fps = scene_loader.fps

# save the frames
frames_out_path = save_scene_path / "Frames"
create_empty_folder(frames_out_path, save_overwrite=True)


next_seg_idx = -1  # index of the current out-of-view segment

# start the new video from the first in-view frame
start_frame = is_in_view.argmin()

frames = []
vis_frames = []

print("Creating new video...")
for i in range(n_frames_new):
    print(f"i_frame={i}/{n_frames_new}", end="\r", flush=True)
    i_frame = new_vid_frame_inds[i]  # index of the current frame in the original video
    frame = scene_loader.get_frame_at_index(i_frame, frame_type="full")
    frames.append(frame)
    vis_frame = np.copy(frame)
    # draw the algorithm view circle
    vis_frame = draw_alg_view_in_the_full_frame(vis_frame, scene_loader)
    # draw bounding boxes for the tracks in the algorithm view (in the full frame)
    alg_view_tracks = detections_tracker.get_tracks_in_frame(i_frame, frame_type="alg_view")
    for track_id, cur_track in alg_view_tracks.items():
        vis_frame = draw_track_box_on_frame(
            rgb_frame=vis_frame,
            track=cur_track,
            track_id=track_id,
            alg_view_cropper=alg_view_cropper,
            convert_from_alg_view_to_full=True,
        )
    vis_frames.append(vis_frame)

# save the videos
save_video_from_frames_list(save_path=save_scene_path / "NewVideo", frames=frames, fps=fps)
save_video_from_frames_list(save_path=save_scene_path / "NewVideo_Vis", frames=vis_frames, fps=fps)
