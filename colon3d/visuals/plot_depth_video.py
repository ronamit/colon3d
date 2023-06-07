from pathlib import Path

import cv2
import numpy as np

from colon3d.general_util import save_video_from_func

# --------------------------------------------------------------------------------------------------------------------


def plot_depth_video(depth_frames: np.ndarray, fps: float, save_path: Path):
    n_frames = depth_frames.shape[0]

    def get_depth_frame(i):
        heatmap = cv2.applyColorMap(np.array(depth_frames[i], dtype=np.uint8), cv2.COLORMAP_JET)
        return heatmap

    save_video_from_func(save_path=save_path, make_frame=get_depth_frame, n_frames=n_frames, fps=fps)


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
