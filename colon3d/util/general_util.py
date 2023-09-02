import argparse
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib import font_manager
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torch.backends import cudnn

# --------------------------------------------------------------------------------------------------------------------


def bool_arg(v):
    # from https://stackoverflow.com/questions/60999816/argparse-not-parsing-boolean-arguments
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


# --------------------------------------------------------------------------------------------------------------------
def val_to_yaml_format(v):
    if isinstance(v, Path):
        return path_to_str(v)
    if isinstance(v, np.float64 | np.float32):
        return float(v)
    return v


# ----------------------------------------------------------------------------------------------------------------------
def save_dict_to_yaml(save_path: Path, dict_to_save: dict):
    save_dict_strs = {k: val_to_yaml_format(v) for k, v in dict_to_save.items()}
    with save_path.open("w") as f:
        yaml.dump(save_dict_strs, f)
    print(f"Saved dict to {save_path}")


# --------------------------------------------------------------------------------------------------------------------


def get_git_version_link():
    try:
        # Get the Git commit hash
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")

        # Get the remote URL of the repository
        remote_url = subprocess.check_output(["git", "config", "--get", "remote.origin.url"]).strip().decode("utf-8")

        # Generate the Git link
        git_link = remote_url.replace(".git", "/commit/") + commit_hash

    except Exception as e:  # noqa: BLE001
        print("Error: Failed to retrieve Git information.")
        print(e)
    return git_link


# --------------------------------------------------------------------------------------------------------------------


def get_run_file_and_args():
    # Get the run file name and arguments use to run the script
    run_file = sys.argv[0]
    run_args = sys.argv[1:]
    return run_file, run_args


# --------------------------------------------------------------------------------------------------------------------


def save_run_info(save_path: Path):
    git_version_link = get_git_version_link()
    run_file, run_args = get_run_file_and_args()
    # save the args + git link  as yaml file
    info_dict = {
        "run_file": to_str(run_file),
        "run_args": to_str(run_args),
        "git_version": git_version_link,
        "time_now": get_time_now_str(),
        "original_save_path": to_str(save_path),
    }
    run_info_path = save_path / "run_info.csv"
    # save the info_dict as a new line in the csv file, or create the file if it does not exist
    pd.DataFrame(info_dict, index=[0]).to_csv(run_info_path, mode="a", header=not run_info_path.exists(), index=False)


# --------------------------------------------------------------------------------------------------------------------


def set_rand_seed(seed):
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


# --------------------------------------------------------------------------------------------------------------------


class ArgsHelpFormatter(
    argparse.RawTextHelpFormatter,
    argparse.ArgumentDefaultsHelpFormatter,
):
    # https://stackoverflow.com/a/68260107/10892246
    pass


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


def save_video_from_func(save_path: Path, make_frame, n_frames: int, fps: float):
    """Saves a video from a function that generates the frames.
    Args:
        save_path: the path to save the video to
        make_frame (function): a function that gets the frame number and returns an RGB frame as a numpy array [H, W, C]
        n_frames (int): the number of frames
        fps: the frames per second
    """
    file_path = str(save_path) + ".mp4"
    frame = make_frame(0)
    height, width = frame.shape[:2]
    print(f"Saving video to {file_path}...")
    if save_path.exists():
        save_path.unlink()
    vid_writer = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for i_frame in range(n_frames):
        # print(f"Saving frame {i_frame + 1}/{n_frames}...", end="\r")
        frame = make_frame(i_frame)
        # change RGB to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        vid_writer.write(frame)
    print(f"Video saved to {file_path}")


# --------------------------------------------------------------------------------------------------------------------


def save_video_from_frames_list(save_path: Path, frames: list, fps: float):
    def make_frame(i_frame) -> np.ndarray:
        return frames[i_frame]

    save_video_from_func(save_path, make_frame, len(frames), fps)
    

# --------------------------------------------------------------------------------------------------------------------


def save_depth_video(depth_frames: np.ndarray, fps: float, save_path: Path):
    n_frames = depth_frames.shape[0]

    def get_depth_frame(i):
        heatmap = cv2.applyColorMap(np.array(depth_frames[i], dtype=np.uint8), cv2.COLORMAP_JET)
        return heatmap

    save_video_from_func(save_path=save_path, make_frame=get_depth_frame, n_frames=n_frames, fps=fps)


# --------------------------------------------------------------------------------------------------------------------
def colors_platte(i_color: int | None = None, color_name: str | None = None):
    # source: https://www.rapidtables.com/web/color/RGB_Color.html
    # RGB colors
    colors_list = [
        ("lime", (0, 255, 0)),
        ("dodger_blue", (30, 144, 255)),
        ("green", (0, 128, 0)),
        ("magenta", (255, 0, 255)),
        ("cyan", (0, 255, 255)),
        ("purple", (128, 0, 128)),
        ("azure", (255, 255, 240)),
        ("blue", (0, 0, 255)),
        ("red", (255, 0, 0)),
        ("yellow", (255, 255, 0)),
        ("white", (255, 255, 255)),
        ("gray", (128, 128, 128)),
    ]
    if color_name is not None:
        return next(c[1] for c in colors_list if c[0] == color_name)
    return colors_list[i_color % len(colors_list)][1]


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


def put_unicode_text_on_img(img, text, pos, font_size, fill_color, stroke_width=1, stroke_fill="black"):
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


def get_time_now_str(tz=None):
    time_now = datetime.now(tz=timezone.utc) if tz is None else datetime.now(tz=tz)
    time_str = f"{time_now.year}-{time_now.month}-{time_now.day}--{time_now.hour}-{time_now.minute}-{time_now.second}"
    return time_str


# --------------------------------------------------------------------------------------------------------------------


def convert_sec_to_str(seconds):
    minutes, sec = divmod(seconds, 60)
    hour, minutes = divmod(minutes, 60)
    return "%d:%02d:%02d" % (hour, minutes, sec)


# --------------------------------------------------------------------------------------------------------------------


def create_empty_folder(folder_path: Path, save_overwrite: bool = False):
    if folder_path.exists() and not save_overwrite:
        return False
    # remove the folder if it already exists
    if folder_path.exists():
        shutil.rmtree(folder_path.resolve())
    folder_path.mkdir(parents=True, exist_ok=True)
    save_run_info(folder_path)
    return True


# --------------------------------------------------------------------------------------------------------------------


def create_folder_if_not_exists(folder_path: Path):
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)
        save_run_info(folder_path)
    return folder_path


# --------------------------------------------------------------------------------------------------------------------


def path_to_str(path: Path):
    return str(path.resolve())


# --------------------------------------------------------------------------------------------------------------------


def get_most_common_values(array, num_values=5):
    """
    Finds the most common values and their count in a NumPy array.

    Args:
    array: A NumPy array.

    Returns:
    A tuple of two NumPy arrays, the first containing the most common values,
    and the second containing their counts.
    """

    # Get the unique values and their counts.
    vals, counts = np.unique(array, return_counts=True)

    # Sort the counts array by descending order.
    sort_idx = np.argsort(-counts)

    return vals[sort_idx][:num_values], counts[sort_idx][:num_values]


# ------------------------------------------------------------


def to_str(a):
    if isinstance(a, Path):
        return str(a.resolve())
    if isinstance(a, torch.Tensor):
        return to_str(a.detach().cpu().numpy())
    if isinstance(a, tuple):
        return "(" + (", ".join([to_str(val) for val in a])) + ")"
    if isinstance(a, list):
        return "[" + (", ".join([to_str(val) for val in a])) + "]"
    if isinstance(a, dict):
        return "{" + (", ".join([f"{key}:{to_str(val)}" for key, val in a.items()])) + "}"
    if isinstance(a, np.ndarray):
        return np.array2string(a, separator=",", precision=2, suppress_small=True)
    if isinstance(a, float):
        return f"{a:.2f}"
    return str(a)


# ------------------------------------------------------------


def save_rgb_image(img: np.ndarray, save_path: Path):
    cv2.imwrite(path_to_str(save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
# ------------------------------------------------------------

def load_rgb_image(img_path: Path):
    frame = cv2.imread(path_to_str(img_path))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return  frame
# ------------------------------------------------------------


def compute_metrics_statistics(metrics_table: pd.DataFrame):
    numeric_columns = metrics_table.select_dtypes(include=[np.number]).columns
    metrics_summary = {}
    for col in numeric_columns:
        mean_val = np.nanmean(metrics_table[col])  # ignore nan values
        std_val = np.nanstd(metrics_table[col])
        n_scenes = np.sum(~np.isnan(metrics_table[col]))
        confidence_interval = 1.96 * std_val / np.sqrt(max(n_scenes, 1))  # 95% confidence interval
        metrics_summary[col] = f"{mean_val:.2f}({confidence_interval:.2f})"
    return metrics_summary


# ------------------------------------------------------------


def save_unified_results_table(base_results_path: Path):
    """save unified results table for all the results subfolders of base_results_path"""
    unified_results_table = pd.DataFrame()
    for results_path in base_results_path.glob("*"):
        if not results_path.is_dir():
            continue
        metrics_table_path = results_path / "err_table.csv"
        if not metrics_table_path.exists():
            continue
        metrics_table = pd.read_csv(metrics_table_path)
        # compute statistics over all scenes
        metrics_summary = compute_metrics_statistics(metrics_table)
        # add the run-name column
        metrics_summary = {"run_name": results_path.name} | metrics_summary
        metrics_summary_df = pd.DataFrame(metrics_summary, index=[0])
        # add the run name to the results table:
        unified_results_table = pd.concat([unified_results_table, metrics_summary_df], axis=0)
    # save the unified results table:
    file_path = base_results_path / "unified_results_table.csv"
    unified_results_table.to_csv(file_path, encoding="utf-8", index=False)
    print(f"Saved unified results table to {file_path}")


# --------------------------------------------------------------------------------------------------------------------


def delete_incomplete_run_dirs(base_results_path: Path):
    for results_path in base_results_path.glob("*"):
        if results_path.is_dir():
            cur_result_path = results_path / "metrics_summary.csv"
            if not cur_result_path.exists():
                print(f"Results dir {results_path} does not contain metrics_summary.csv, deleting it ..")
                shutil.rmtree(results_path)


# --------------------------------------------------------------------------------------------------------------------


def to_path(path):
    return Path(path) if path is not None else None


# --------------------------------------------------------------------------------------------------------------------


def replace_keys(d: dict, old_key, new_key):
    if old_key in d:
        d[new_key] = d.pop(old_key)
    return d
# --------------------------------------------------------------------------------------------------------------------
