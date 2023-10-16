from pathlib import Path

import attrs
import yaml

from colon_nav.util.general_util import save_dict_to_yaml

# ---------------------------------------------------------------------------------------------------------------------


@attrs.define
class ModelInfo:
    depth_model_name: str
    egomotion_model_name: str
    ref_frame_shifts: list[int]  # The time shifts of the reference frames w.r.t. the target frame
    depth_calib_type: str = "none"
    depth_calib_a: float = 1.0
    depth_calib_b: float = 0.0
    # lower bound to clip the the z-depth estimation (units: mm). If None - then no lower bound is used:
    depth_lower_bound: float | None = None
    # upper bound to clip the the z-depth estimation (units: mm). If None - then no upper bound is used:
    depth_upper_bound: float | None = None
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

    model_info_dict = attrs.asdict(model_info)

    save_dict_to_yaml(save_path=model_info_path, dict_to_save=model_info_dict)
    print(f"Saved model info to {model_info_path}")
    print("Model info:", model_info)


# ---------------------------------------------------------------------------------------------------------------------


def load_model_model_info(model_path: Path) -> ModelInfo:
    model_info_path = model_path / "model_info.yaml"
    if not model_info_path.exists():
        raise FileNotFoundError(f"Model info file {model_info_path} does not exist")
    with model_info_path.open("r") as f:
        model_info = yaml.safe_load(f)
    return ModelInfo(**model_info)


# ---------------------------------------------------------------------------------------------------------------------
