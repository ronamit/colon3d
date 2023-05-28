import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from cv2 import cv2
from dateutil import tz
from matplotlib import font_manager
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------------------------------------------------


def find_first_true(arr, start_ind=0, stop_ind=-1):
    for i in range(start_ind, len(arr)):
        if (i == stop_ind) or arr[i]:
            return i
    return None


# --------------------------------------------------------------------------------------------------------------------


def find_between_str(s, before_str, after_str):
    """Finds the string between two strings in a file.

    Args:
        s (str): the string to search in
        before_str (str): the string before the desired string
        after_str (str): the string after the desired string

    Returns:
        str: the desired string
    """
    start_index = s.find(before_str)
    if start_index == -1:
        return None
    start_index += len(before_str)
    end_index = s.find(after_str, start_index)
    if end_index == -1:
        return None
    return s[start_index:end_index]


# --------------------------------------------------------------------------------------------------------------------
def find_in_file_between_str(file_path, before_str, after_str, line_prefix=None):
    """Finds the string between two strings in a file.

    Args:
        file_path (str): the path to the file
        before_str (str): the string before the desired string
        after_str (str): the string after the desired string
        line_prefix(str): if not None, the line must start with this prefix (default: None

    Returns:
        str: the desired string
    """
    matches = []
    file_path = Path(file_path)
    with file_path.open() as file:
        lines = file.readlines()
        for line in lines:
            if line_prefix is not None and not line.startswith(line_prefix):
                continue
            match = find_between_str(line, before_str, after_str)
            if match is not None:
                matches.append(match)
    assert len(matches) == 1, f"Found {len(matches)} matches for {before_str} and {after_str} in {file_path}"
    return matches[0]


# --------------------------------------------------------------------------------------------------------------------


def save_video_from_func(save_path: Path, file_name: str, make_frame, n_frames: int, fps: float, backend="cv2"):
    """Saves a video from a function that generates the frames.
    Args:
        save_path: the path to save the video to
        file_name: the name of the video file
        make_frame (function): a function that gets the frame number and returns the frame as a numpy array [H, W, C]
        n_frames (int): the number of frames
        fps: the frames per second
    Note: for faster video creation, see https://github.com/opencv/opencv/wiki/Video-IO-hardware-acceleration
    """
    print(f"Creating video: {file_name}...")
    file_path = str(Path(save_path) / file_name)
    if backend == "cv2":
        frame = make_frame(0)
        height, width = frame.shape[:2]
        vid_writer = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc("m", "p", "4", "v"), fps, (width, height))
        for i_frame in range(n_frames):
            print(f"Saving frame {i_frame + 1}/{n_frames}...", end="\r")
            frame = make_frame(i_frame)
            vid_writer.write(frame)
        vid_writer.release()
    # elif backend == "moviepy_nvidia":
    #     import moviepy.editor as mpy
    #     duration = n_frames / fps
    #     animation_clip = mpy.VideoClip(make_frame, duration=duration)
    #     animation_clip.write_videofile(str(save_path / (file_name + ".mp4")), fps=fps, threads=5, codec="h264_nvenc")
    else:
        raise ValueError("Unknown backend")
    print(f"Video saved to {file_path}")


# --------------------------------------------------------------------------------------------------------------------


def save_video(save_path: Path, file_name: str, frames: list, fps: float):
    def make_frame(i_frame) -> np.ndarray:
        return frames[i_frame]

    save_video_from_func(save_path, file_name, make_frame, len(frames), fps)


# --------------------------------------------------------------------------------------------------------------------
def colors_platte(i_color, color_type="BGR"):
    # source: https://www.rapidtables.com/web/color/RGB_Color.html
    # BGR colors
    colors = [
        (255, 144, 30),  # dodger blue
        (0, 255, 0),  # lime
        (255, 0, 255),  # magenta
        (0, 255, 255),  # cyan
        (255, 255, 0),  # yellow
        (128, 0, 128),  # purple
        (255, 255, 240),  # azure
        (0, 128, 0),  # green
    ]

    color = colors[i_color % len(colors)]
    if color_type == "BGR":
        return color
    if color_type == "RGB":
        return color[::-1]
    raise ValueError("type must be BGR or RGB")


# --------------------------------------------------------------------------------------------------------------------


def coord_to_cv2kp(coord):
    # note: if negative coordenates are given, they are set to 0 (otherwise, cv2.KeyPoint will give junk)
    x, y = coord
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    return cv2.KeyPoint(x, y, size=1)


# --------------------------------------------------------------------------------------------------------------------


def save_plot_and_close(save_path):
    plt.savefig(save_path)
    plt.cla()
    plt.clf()
    plt.close()
    print(f"Plot saved to {save_path}")


# --------------------------------------------------------------------------------------------------------------------


def is_point_in_img(point, image):
    return point[0] >= 0 and point[0] < image.shape[1] and point[1] >= 0 and point[1] < image.shape[0]


# --------------------------------------------------------------------------------------------------------------------


def is_equal_dicts(dict1, dict2):
    """Return whether two dicts of arrays are exactly equal."""
    if dict1.keys() != dict2.keys():
        return False
    return all(np.array_equal(dict1[key], dict2[key]) for key in dict1)


# --------------------------------------------------------------------------------------------------------------------


def put_unicode_text_on_img(img, text, pos, font_size, fill_color, stroke_width, stroke_fill):
    """Put unicode text on image (note that OpeCV does not support unicode text, so we use PIL)

    Args:
        img (np.array): image to put text on
        text (str): text to put on image
        pos (tuple): position of text on image
        font_size (int): font size
        fill_color (tuple): color of text (BGR)
        stroke_width (int): stroke width
        stroke_fill (tuple): stroke color
    """
    # Make into PIL Image
    im_p = Image.fromarray(img)
    # Get a drawing context
    draw = ImageDraw.Draw(im_p)
    # Find the path for a specific font:
    font_path = font_manager.findfont("DejaVu Sans")
    font = ImageFont.truetype(font=font_path, size=round(font_size))
    draw.text(xy=pos, text=text, fill=fill_color, font=font, stroke_width=stroke_width, stroke_fill=stroke_fill)
    # Convert back to OpenCV image and save
    result_o = np.array(im_p)
    return result_o


# --------------------------------------------------------------------------------------------------------------------


def clip_arrow(tip, base, max_len, lower_lim_x, lower_lim_y, upper_lim_x, upper_lim_y):
    vec = tip - base
    vec_len = np.linalg.norm(vec)
    if vec_len > max_len:
        vec = max_len * vec / vec_len
    tip = base + vec
    tip = np.maximum(tip, np.array([lower_lim_x, lower_lim_y]))
    tip = np.minimum(tip, np.array([upper_lim_x, upper_lim_y]))
    return tip


# --------------------------------------------------------------------------------------------------------------------


class Tee:
    """
    Context manager that copies stdout and any exceptions to a log fil/e
    It also writes any exceptions to the file.
    Source: https://stackoverflow.com/a/57008707
    """

    def __init__(self, filepath: Path):
        if not (filepath.parent).exists():
            (filepath.parent).mkdir(parents=True)
        self.file = filepath.open("w")
        self.stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self

    def __exit__(self, exc_type, exc_value, tb):
        sys.stdout = self.stdout
        if exc_type is not None and exc_value is not None and tb is not None:
            self.file.write(traceback.format_exc())
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()


# --------------------------------------------------------------------------------------------------------------------


def get_time_now_str():
    local_time = datetime.now(tz.gettz())
    return local_time.strftime("%Y-%m-%d %H:%M %Z %z")


# --------------------------------------------------------------------------------------------------------------------


def convert_sec_to_str(seconds):
    minutes, sec = divmod(seconds, 60)
    hour, minutes = divmod(minutes, 60)
    return "%d:%02d:%02d" % (hour, minutes, sec)


# --------------------------------------------------------------------------------------------------------------------


def create_empty_folder(folder_path: Path, ask_overwrite: bool = False):
    if folder_path.exists():
        if ask_overwrite:
            print(f"Folder {folder_path} already exists.. overwrite it? (y/n): ", end="")
            user_answer = input()
            delete_dir = user_answer in ["y", "Y"]
        else:
            print(f"Folder {folder_path} already exists.. overwriting it....")
            delete_dir = True
        if delete_dir:
            shutil.rmtree(folder_path)
        else:
            print("Exiting...")
            sys.exit(0)
    folder_path.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------------------------------------


def path_to_str(path: Path):
    return str(path.resolve())


# --------------------------------------------------------------------------------------------------------------------
