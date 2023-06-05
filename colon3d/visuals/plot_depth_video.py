from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------------------------------------------------


def plot_depth_video(depth_frames: np.ndarray, fps: float, save_path: Path):
    n_frames = depth_frames.shape[0]
    save_path_str = str(save_path.resolve())

    video_writer = cv2.VideoWriter(
        save_path_str,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (depth_frames.shape[1], depth_frames.shape[2]),
    )

    # Loop over frames and create heatmaps.
    for i in range(n_frames):
        heatmap = cv2.applyColorMap(np.array(depth_frames[i], dtype=np.uint8), cv2.COLORMAP_JET)
        video_writer.write(heatmap)

    # Close the video writer.
    video_writer.release()
    print(f"Saved depth video to {save_path_str}")


# --------------------------------------------------------------------------------------------------------------------


# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import matplotlib.animation as animation

# img = [] # some array of images
# frames = [] # for storing the generated images
# fig = plt.figure()
# for i in xrange(6):
#     frames.append([plt.imshow(img[i], cmap=cm.Greys_r,animated=True)])

# ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
#                                 repeat_delay=1000)
# # ani.save('movie.mp4')
