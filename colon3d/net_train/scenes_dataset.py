import random
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils import data

from colon3d.net_train import endosfm_transforms, md2_transforms
from colon3d.net_train.train_utils import DatasetMeta
from colon3d.util.data_util import get_all_scenes_paths_in_dir
from colon3d.util.torch_util import to_default_type, to_torch

# ---------------------------------------------------------------------------------------------------------------------


class ScenesDataset(data.Dataset):
    # ---------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        scenes_paths: list,
        feed_height: int,
        feed_width: int,
        model_name: str,
        dataset_type: str,
        subsample_min: int = 1,
        subsample_max: int = 3,
        ref_frame_shifts_base: tuple = (-1, 1),
        n_sample_lim: int = 0,
        load_gt_depth: bool = False,
        load_gt_pose: bool = False,
        plot_example_ind: int | None = None,
    ):
        r"""Initialize the DatasetLoader class
        Args:
            scenes_paths (list): List of paths to the scenes
            feed_height (int): The height of the input images to the network
            feed_width (int): The width of the input images to the network
            load_gt_depth (bool): Whether to add the depth map of the target frame to each sample (default: False)
            load_gt_pose (bool): Whether to add the ground-truth pose change between from target to the reference frames,  each sample (default: False)
            transforms: transforms to apply to each sample  (in order)
            subsample_min (int): Minimum subsample factor to set the frame number between frames in the example.
            subsample_max (int): Maximum subsample factor to set the frame number between frames in the example.
            n_sample_lim (int): Limit the number of samples to load (for debugging) if 0 then no limit
        Notes:
            for each training example, we randomly a subsample factor to set the frame number between frames in the example (to get wider range of baselines \ ego-motions between the frames)
        """
        self.scenes_paths = scenes_paths
        self.feed_height = feed_height
        self.feed_width = feed_width
        self.model_name = model_name
        self.dataset_type = dataset_type
        self.subsample_min = subsample_min
        self.subsample_max = subsample_max
        self.ref_frame_shifts_base = ref_frame_shifts_base
        self.num_input_images = (
            len(ref_frame_shifts_base) + 1
        )  # number of frames in each sample (target + reference frames)
        self.load_gt_depth = load_gt_depth
        self.load_gt_pose = load_gt_pose
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

        self.dataset_meta = DatasetMeta(
            feed_height=self.feed_height,
            feed_width=self.feed_width,
            load_gt_depth=self.load_gt_depth,
            load_gt_pose=self.load_gt_pose,
            num_input_images=self.num_input_images,
        )

        # set data transforms
        if model_name == "EndoSFM":
            if dataset_type == "train":
                self.transform = endosfm_transforms.get_train_transform(dataset_meta=self.dataset_meta)
            elif dataset_type == "val":
                self.transform = endosfm_transforms.get_val_transform(dataset_meta=self.dataset_meta)
            else:
                raise ValueError(f"Unknown dataset type: {dataset_type}")

        elif model_name == "MonoDepth2":
            n_scales_md2 = 4
            if dataset_type == "train":
                self.transform = md2_transforms.get_train_transform(
                    dataset_meta=self.dataset_meta,
                    n_scales=n_scales_md2,
                )
            elif dataset_type == "val":
                self.transform = md2_transforms.get_val_transform(dataset_meta=self.dataset_meta, n_scales=n_scales_md2)
            else:
                raise ValueError(f"Unknown dataset type: {dataset_type}")
        else:
            raise ValueError(f"Unknown method: {model_name}")

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

        # "frame_inds" = list of the indices of the frames in the example
        # The first frame is always the target frame, the rest are the reference frames
        frame_inds = [target_frame_idx]
        frame_shifts = [0]

        # Add the reference frames indices
        for ref_shift_base in self.ref_frame_shifts_base:
            # randomly choose the subsample factor
            subsample_factor = torch.randint(self.subsample_min, self.subsample_max + 1, (1,)).item()
            # load the reference frame
            shift = subsample_factor * ref_shift_base
            ref_frame_idx = target_frame_idx + shift
            frame_inds.append(ref_frame_idx)
            frame_shifts.append(shift)

        # get the camera intrinsics matrix
        with (scene_path / "meta_data.yaml").open() as file:
            scene_metadata = yaml.load(file, Loader=yaml.FullLoader)
            intrinsics_orig = get_camera_matrix(scene_metadata)

        sample["K"] = to_torch(intrinsics_orig)

        # load the RGB images
        for i, frame_idx in enumerate(frame_inds):
            frame_path = scene_frames_paths[frame_idx]
            # Load with PIL
            sample[("color", i)] = load_img_and_resize(frame_path, height=self.feed_height, width=self.feed_width)

        # load the ground-truth depth map and the ground-truth pose
        if self.load_gt_depth or self.load_gt_pose:
            with h5py.File((scene_path / "gt_3d_data.h5").resolve(), "r") as h5f:
                if self.load_gt_depth:
                    for i, frame_idx in enumerate(frame_inds):
                        depth_gt = to_default_type(h5f["z_depth_map"][frame_idx], num_type="float_m")
                        sample[("depth_gt", i)] = depth_gt
                if self.load_gt_pose:
                    for i, frame_idx in enumerate(frame_inds):
                        abs_pose = to_default_type(h5f["cam_poses"][frame_idx])
                        sample[("abs_pose", i)] = abs_pose
        # apply the transform
        if self.transform:
            sample = self.transform(sample)
        return sample

    # ---------------------------------------------------------------------------------------------------------------------

    def get_scene_metadata(self, i_scene: int = 0) -> dict:
        scene_path = self.scenes_paths[i_scene]
        with (scene_path / "meta_data.yaml").open() as file:
            metadata = yaml.load(file, Loader=yaml.FullLoader)
        return metadata


# ---------------------------------------------------------------------------------------------------------------------


def get_camera_matrix(scene_metadata: dict) -> np.ndarray:
    cam_K = np.zeros((3, 3), dtype=np.float32)
    cam_K[0, 0] = scene_metadata["fx"]
    cam_K[1, 1] = scene_metadata["fy"]
    cam_K[0, 2] = scene_metadata["cx"]
    cam_K[1, 2] = scene_metadata["cy"]
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


def load_img_and_resize(img_path: Path, height: int, width: int):
    """Load an image and resize it".
    Args:
        img_path (str): Path to the image
        new_size (tuple): The new size of the image (height, width)
    Returns:
        PIL.Image: The loaded and resized image
    """
    img = Image.open(img_path)
    img = img.resize((width, height), Image.ANTIALIAS)
    return img


# ---------------------------------------------------------------------------------------------------------------------
