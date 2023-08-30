import random
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from PIL import Image
import matplotlib.pyplot as plt
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
        transform: Compose | None = None,
        subsample_min: int = 1,
        subsample_max: int = 20,
        n_sample_lim: int = 0,
        load_target_depth: bool = False,
        plot_example_ind: int | None = None,
    ):
        r"""Initialize the DatasetLoader class
        Args:
            scenes_paths (list): List of paths to the scenes
            load_tgt_depth (bool): Whether to add the depth map of the target frame to each sample (default: False)
            transforms: transforms to apply to each sample  (in order)
            subsample_min (int): Minimum subsample factor to set the frame number between frames in the example.
            subsample_max (int): Maximum subsample factor to set the frame number between frames in the example.
            n_sample_lim (int): Limit the number of samples to load (for debugging) if 0 then no limit
        Notes:
            for each training example, we randomly a subsample factor to set the frame number between frames in the example (to get wider range of baselines \ ego-motions between the frames)
        """
        self.scenes_paths = scenes_paths
        self.transform = transform
        self.subsample_min = subsample_min
        self.subsample_max = subsample_max
        self.load_target_depth = load_target_depth
        self.plot_example_ind = plot_example_ind
        assert self.subsample_min <= self.subsample_max
        self.frame_paths_per_scene = []
        self.target_ids = []
        # go over all the scenes in the dataset
        frames_paths_per_scene = []
        n_frames_per_scene = []
        for scene_path in self.scenes_paths:
            frames_paths = [
                frame_path
                for frame_path in (scene_path / "RGB_Frames").iterdir()
                if frame_path.is_file() and frame_path.name.endswith(".png")
            ]
            if n_sample_lim > 0:
                frames_paths = frames_paths[:n_sample_lim]
            self.frame_paths_per_scene.append(frames_paths)
            frames_paths.sort()
            frames_paths_per_scene.append(frames_paths)
            n_frames_per_scene.append(len(frames_paths))

        self.subsample_min = min(self.subsample_min, min(n_frames_per_scene) - 1)
        self.subsample_max = min(self.subsample_max, min(n_frames_per_scene) - 1)

        for i_scene, frames_paths in enumerate(frames_paths_per_scene):
            n_frames = len(frames_paths)
            assert n_frames > 0
            # set the scene index and the target frame index for each sample (later we will set the reference frames indices)
            for i_frame in range(n_frames - self.subsample_max):
                self.target_ids.append({"scene_idx": i_scene, "target_frame_idx": i_frame})

    # ---------------------------------------------------------------------------------------------------------------------

    def __len__(self):
        return len(self.target_ids)

    # ---------------------------------------------------------------------------------------------------------------------

    def __getitem__(self, index: int) -> dict:
        sample = {}
        target_id = self.target_ids[index]
        scene_index = target_id["scene_idx"]
        target_frame_idx = target_id["target_frame_idx"]
        scene_path = self.scenes_paths[scene_index]
        scene_frames_paths = self.frame_paths_per_scene[scene_index]
        sample["target_frame_idx"] = target_frame_idx
        # get the camera intrinsics matrix
        with (scene_path / "meta_data.yaml").open() as file:
            metadata = yaml.load(file, Loader=yaml.FullLoader)
            intrinsics_orig = get_camera_matrix(metadata)

        sample["intrinsics_K"] = to_torch(intrinsics_orig)
        # note that the intrinsics matrix might be changed later by the transform (as needed for some methods)

        # load the target frame
        target_frame_ind = target_id["target_frame_idx"]
        target_frame_path = scene_frames_paths[target_frame_ind]
        # Load with PIL
        sample["target_img"] = Image.open(target_frame_path)

        # randomly choose the subsample factor
        subsample_factor = torch.randint(self.subsample_min, self.subsample_max + 1, (1,)).item()

        # load the reference frame
        ref_frame_idx = target_frame_ind + subsample_factor
        sample["ref_img"] = Image.open(scene_frames_paths[ref_frame_idx])

        if self.load_target_depth:
            with h5py.File((scene_path / "gt_3d_data.h5").resolve(), "r") as h5f:
                target_depth = to_default_type(h5f["z_depth_map"][target_frame_idx], num_type="float_m")
                sample["target_depth"] = target_depth
                
        # apply the transform
        if self.transform:
            sample = self.transform(sample)
            
        # plot example sample for debugging
        if self.plot_example_ind is not None and index == self.plot_example_ind:
            plot_sample(sample)
        return sample

    # ---------------------------------------------------------------------------------------------------------------------

    def get_scene_metadata(self, i_scene: int = 0) -> dict:
        scene_path = self.scenes_paths[i_scene]
        with (scene_path / "meta_data.yaml").open() as file:
            metadata = yaml.load(file, Loader=yaml.FullLoader)
        return metadata


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

def plot_sample(sample: dict):

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(sample["target_img"])
    plt.title("target_img")
    plt.subplot(1, 2, 2)
    plt.imshow(sample["ref_img"])
    plt.title("ref_img")
    plt.show()

# ---------------------------------------------------------------------------------------------------------------------
