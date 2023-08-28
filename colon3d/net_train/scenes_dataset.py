import random
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from imageio import imread
from torch.utils import data
from torchvision.transforms import Compose

from colon3d.util.data_util import get_all_scenes_paths_in_dir
from colon3d.util.torch_util import to_default_type, to_torch

# ---------------------------------------------------------------------------------------------------------------------


class ScenesDataset(data.Dataset):
    # ---------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        scenes_paths: list,
        load_target_depth: bool = False,
        transform: Compose | None = None,
        subsample_min: int = 1,
        subsample_max: int = 20,
    ):
        r"""Initialize the DatasetLoader class
        Args:
            scenes_paths (list): List of paths to the scenes
            load_tgt_depth (bool): Whether to add the depth map of the target frame to each sample (default: False)
            transforms: transforms to apply to each sample  (in order)
            subsample_min (int): Minimum subsample factor to set the frame number between frames in the example.
            subsample_max (int): Maximum subsample factor to set the frame number between frames in the example.
        Notes:
            for each training example, we randomly a subsample factor to set the frame number between frames in the example (to get wider range of baselines \ ego-motions between the frames)
        """
        self.scenes_paths = scenes_paths
        self.load_target_depth = load_target_depth
        self.transform = transform
        self.subsample_min = subsample_min
        self.subsample_max = subsample_max
        assert self.subsample_min <= self.subsample_max
        self.frame_paths_per_scene = []
        self.target_ids = []
        # go over all the scenes in the dataset
        for i_scene, scene_path in enumerate(self.scenes_paths):
            frames_paths = [
                frame_path
                for frame_path in (scene_path / "RGB_Frames").iterdir()
                if frame_path.is_file() and frame_path.name.endswith(".png")
            ]
            self.frame_paths_per_scene.append(frames_paths)
            frames_paths.sort()
            n_frames = len(frames_paths)
            max_tgt_frame_ind = max(0, n_frames - 1 - self.subsample_max)
            # set the scene index and the target frame index for each sample (later we will set the reference frames indices)
            for i_frame in range(max_tgt_frame_ind):
                self.target_ids.append({"scene_idx": i_scene, "target_frame_idx": i_frame})

    # ---------------------------------------------------------------------------------------------------------------------

    def __len__(self):
        return len(self.target_ids)

    # ---------------------------------------------------------------------------------------------------------------------

    def __getitem__(self, index: int) -> dict:
        sample = {}
        target_id = self.target_ids[index]
        scene_index = target_id["scene_idx"]
        scene_path = self.scenes_paths[scene_index]
        scene_frames_paths = self.frame_paths_per_scene[scene_index]

        # get the camera intrinsics matrix
        with (scene_path / "meta_data.yaml").open() as file:
            metadata = yaml.load(file, Loader=yaml.FullLoader)
            intrinsics_orig = get_camera_matrix(metadata)

        sample["intrinsics_K"] = to_torch(intrinsics_orig)
        # note that the intrinsics matrix might be changed later by the transform (as needed for some methods)

        # load the target frame
        target_frame_ind = target_id["target_frame_idx"]
        target_frame_path = scene_frames_paths[target_frame_ind]
        sample["target_img"] = load_as_float(target_frame_path)

        # randomly choose the subsample factor
        subsample_factor = torch.randint(self.subsample_min, self.subsample_max + 1, (1,)).item()

        # load the reference frame
        ref_frame_idx = target_frame_ind + subsample_factor
        sample["ref_img"] = load_as_float(scene_frames_paths[ref_frame_idx])

        if self.load_target_depth:
            # load the depth map of the target frame and return it as part of the sample (As is, without any transformation)
            with h5py.File((scene_path / "gt_3d_data.h5").resolve(), "r") as h5f:
                target_depth = to_default_type(h5f["z_depth_map"][target_frame_ind], num_type="float_m")
                sample["target_depth"] = target_depth

        # apply the transform
        if self.transform:
            sample = self.transform(sample)
        return sample

    # ---------------------------------------------------------------------------------------------------------------------

    def get_scene_metadata(self, i_scene: int) -> dict:
        scene_path = self.scenes_paths[i_scene]
        with (scene_path / "meta_data.yaml").open() as file:
            metadata = yaml.load(file, Loader=yaml.FullLoader)
        return metadata


# ---------------------------------------------------------------------------------------------------------------------


def load_as_float(path):
    return imread(path).astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------------


def get_camera_matrix(metadata: dict) -> np.ndarray:
    cam_K = np.zeros((3, 3), dtype=np.float32)
    cam_K[0, 0] = metadata["fx"]
    cam_K[1, 1] = metadata["fy"]
    cam_K[0, 2] = metadata["cx"]
    cam_K[1, 2] = metadata["cy"]
    cam_K[2, 2] = 1
    return cam_K


# ---------------------------------------------------------------------------------------------------------------------


def get_scenes_dataset_random_split(dataset_path: Path, validation_ratio: float):
    all_scenes_paths = get_all_scenes_paths_in_dir(dataset_path, with_targets=False)
    random.shuffle(all_scenes_paths)
    n_all_scenes = len(all_scenes_paths)
    n_train_scenes = int(n_all_scenes * (1 - validation_ratio))
    n_val_scenes = n_all_scenes - n_train_scenes
    train_scenes_paths = all_scenes_paths[:n_train_scenes]
    val_scenes_paths = all_scenes_paths[n_train_scenes:]
    print(f"Number of training scenes {n_train_scenes}, validation scenes {n_val_scenes}")
    return train_scenes_paths, val_scenes_paths


# ---------------------------------------------------------------------------------------------------------------------
