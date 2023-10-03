from dataclasses import dataclass
from pathlib import Path

import yaml

from colon_nav.util.general_util import save_dict_to_yaml

# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class DatasetMeta:
    feed_height: int  # The height of the input images to the network
    feed_width: int  # The width of the input images to the network
    load_gt_depth: bool  # Whether to add the depth map of the target frame to each sample (default: False)
    load_gt_pose: bool  # Whether to add the ground-truth pose change between from target to the reference frames,  each sample (default: False)
    n_ref_imgs: int  # number of reference frames to use (each sample will have n_ref_imgs frames + 1 target frame)
    ref_frame_shifts: list[int]  # The time shifts of the reference frames w.r.t. the target frame


# ---------------------------------------------------------------------------------------------------------------------


def get_default_model_info(model_name: str) -> dict:
    if model_name in {"EndoSFM", "MonoDepth2"}:
        return {
            "feed_height": 320,  # The image size the network receives as input, in pixels
            "feed_width": 320,
            "num_layers": 18,  # number of ResNet layers in the PoseNet
        }
    raise ValueError(f"Unknown model name: {model_name}")


# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class ModelInfo:
    model_name: str
    n_ref_imgs: int
    feed_height: int
    feed_width: int
    num_layers: int | None
    depth_calib_type: str = "none"
    depth_calib_a: float = 1.0
    depth_calib_b: float = 0.0
    model_description: str = ""


# ---------------------------------------------------------------------------------------------------------------------


def save_model_info(
    save_dir_path: Path,
    model_info: ModelInfo,
    overwrite: bool = True,
):
    model_info_path = save_dir_path / "model_info.yaml"
    if model_info_path.exists() and not overwrite:
        print(f"Model info file {model_info_path} already exists, overwriting")
        model_info_path.unlink()

    model_info_dict = model_info.__dict__

    save_dict_to_yaml(save_path=model_info_path, dict_to_save=model_info_dict)


# ---------------------------------------------------------------------------------------------------------------------


def load_model_model_info(path: Path) -> ModelInfo:
    model_info_path = path / "model_info.yaml"
    if not model_info_path.exists():
        raise FileNotFoundError(f"Model info file {model_info_path} does not exist")
    with model_info_path.open("r") as f:
        model_info = yaml.load(f, Loader=yaml.FullLoader)
    return ModelInfo(**model_info)


# ---------------------------------------------------------------------------------------------------------------------
