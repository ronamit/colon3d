from pathlib import Path

import numpy as np
import torch
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from colon3d.util.general_util import save_dict_to_yaml

# ---------------------------------------------------------------------------------------------------------------------


def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
    # Construct the list colormap, with interpolated values for higer resolution
    # For a linear segmented colormap, you can just specify the number of point in
    # cm.get_cmap(name, lutsize) with the parameter lutsize
    x = np.linspace(0, 1, low_res_cmap.N)
    low_res = low_res_cmap(x)
    new_x = np.linspace(0, max_value, resolution)
    high_res = np.stack([np.interp(new_x, x, low_res[:, i]) for i in range(low_res.shape[1])], axis=1)
    return ListedColormap(high_res)


# ---------------------------------------------------------------------------------------------------------------------


def opencv_rainbow(resolution=1000):
    # Construct the opencv equivalent of Rainbow
    opencv_rainbow_data = (
        (0.000, (1.00, 0.00, 0.00)),
        (0.400, (1.00, 1.00, 0.00)),
        (0.600, (0.00, 1.00, 0.00)),
        (0.800, (0.00, 0.00, 1.00)),
        (1.000, (0.60, 0.00, 1.00)),
    )

    return LinearSegmentedColormap.from_list("opencv_rainbow", opencv_rainbow_data, resolution)


COLORMAPS = {
    "rainbow": opencv_rainbow(),
    "magma": high_res_colormap(cm.get_cmap("magma")),
    "bone": cm.get_cmap("bone", 10000),
}

# ---------------------------------------------------------------------------------------------------------------------


def tensor2array(tensor, max_value=None, colormap="rainbow"):
    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        norm_array = tensor.squeeze().numpy() / max_value
        array = COLORMAPS[colormap](norm_array).astype(np.float32)
        array = array.transpose(2, 0, 1)

    elif tensor.ndimension() == 3:
        assert tensor.size(0) == 3
        array = 0.45 + tensor.numpy() * 0.225
    return array


# ---------------------------------------------------------------------------------------------------------------------


def save_checkpoint(
    save_path: Path,
    dispnet_state,
    exp_pose_state,
    is_best: bool,
    scene_metadata=None,
):
    file_prefixes = ["DispNet", "PoseNet"]
    states = [dispnet_state, exp_pose_state]
    for prefix, state in zip(file_prefixes, states, strict=True):
        save_model_path = save_path / f"{prefix}_checkpoint.pt"
        torch.save(state, save_model_path)
        print(f"Checkpoint saved to {save_model_path}")
        if is_best:
            save_best_path = save_path / f"{prefix}_best.pt"
            torch.save(state, save_best_path)
            print(f"Best checkpoint so-far saved to {save_model_path}")

    # save the scene metadata as yaml file
    if scene_metadata is not None:
        save_dict_to_yaml(save_path=save_path / "scene_metadata.yaml", dict_to_save=scene_metadata)


# ---------------------------------------------------------------------------------------------------------------------


def save_model_info(
    save_dir_path: Path,
    num_layers: int,
    scene_metadata: dict,
    overwrite: bool,
    extra_info: dict | None = None,
):
    model_info_path = save_dir_path / "model_info.yaml"
    if model_info_path.exists() and not overwrite:
        print(f"Model info file {model_info_path} already exists, skipping")

    # save an updated model_info.yaml file:
    model_info = {
        "ResNet_layers": num_layers,
        "frame_height": scene_metadata["frame_height"], # the height of the frames in the data we trained with
        "frame_width": scene_metadata["frame_width"], # the width of the frames in the data we trained with
        "fx": scene_metadata["fx"],
        "fy": scene_metadata["fy"],
        "cx": scene_metadata["cx"],
        "cy": scene_metadata["cy"],
        "distort_pram": scene_metadata["distort_pram"],
        "fps": scene_metadata["fps"],
    }
    if extra_info is not None:
        model_info.update(extra_info)
        save_dict_to_yaml(save_path=model_info_path, dict_to_save=model_info)


# ---------------------------------------------------------------------------------------------------------------------
