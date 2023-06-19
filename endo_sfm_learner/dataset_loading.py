import random
from pathlib import Path

import numpy as np
import torch
from imageio import imread
from torch.utils import data
from colon3d.general_util import create_empty_folder, get_time_now_str, set_rand_seed

# ---------------------------------------------------------------------------------------------------------------------

def load_as_float(path):
    return imread(path).astype(np.float32)

# ---------------------------------------------------------------------------------------------------------------------

class TraintLoader(data.Dataset):
    # ---------------------------------------------------------------------------------------------------------------------
    def __init__(self,dataset_path: Path, seed: int, sample_frames_len: int, transforms: list | None = None):
        """ Initialize the DatasetLoader class
        Args:
            dataset_path (Path): Path to the dataset
            seed (int): Seed for the random number generator
            sample_frames_len (int): Number of consecutive frames in each sample (the reference frame is in the middle)
            transforms (list | None): List of transforms to apply to each sample (default: None)
                transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
        """
        self.dataset_path = dataset_path
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.sample_frames_len = sample_frames_len
        self.transforms = transforms
        self.samples = []
        # go over all the scenes in the dataset
        
        

    # ---------------------------------------------------------------------------------------------------------------------
    
    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample["target_img"])
        ref_imgs = [load_as_float(ref_img) for ref_img in sample["ref_imgs"]]
        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img, *ref_imgs], np.copy(sample["intrinsics_mat"]))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample["intrinsics_mat"])
        return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics)
    
    # ---------------------------------------------------------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)
        
# ---------------------------------------------------------------------------------------------------------------------

class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
    root/scene_1/0000000.jpg
    root/scene_1/0000001.jpg
    ..
    root/scene_1/cam.txt
    root/scene_2/0000000.jpg
    .
    transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, seed=None, train=True, sequence_length=3, img_transforms=None, skip_frames=1):
        set_rand_seed(seed)
        self.root = Path(root)
        scene_list_path = self.root / "train.txt" if train else self.root / "val.txt"
        self.scenes = [self.root / folder[:-1] for folder in scene_list_path.open()]
        self.transform = transform
        self.k = skip_frames
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        # k skip frames
        sequence_set = []
        demi_length = (sequence_length - 1) // 2
        shifts = list(range(-demi_length * self.k, demi_length * self.k + 1, self.k))
        shifts.pop(demi_length)
        for scene in self.scenes:
            intrinsics = np.genfromtxt(scene / "cam.txt").astype(np.float32).reshape((3, 3))
            imgs = sorted(scene.files("*.jpg"))

            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length * self.k, len(imgs) - demi_length * self.k):
                sample = {"intrinsics": intrinsics, "tgt": imgs[i], "ref_imgs": []}
                for j in shifts:
                    sample["ref_imgs"].append(imgs[i + j])
                sequence_set.append(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample["tgt"])
        ref_imgs = [load_as_float(ref_img) for ref_img in sample["ref_imgs"]]
        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img, *ref_imgs], np.copy(sample["intrinsics"]))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample["intrinsics"])
        return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics)

    def __len__(self):
        return len(self.samples)


def crawl_folders(folders_list, dataset="nyu"):
    imgs = []
    depths = []
    for folder in folders_list:
        current_imgs = sorted(folder.files("*.jpg"))
        if dataset == "nyu":
            current_depth = sorted((folder / "depth/").files("*.png"))
        elif dataset == "kitti":
            current_depth = sorted(folder.files("*.npy"))
        imgs.extend(current_imgs)
        depths.extend(current_depth)
    return imgs, depths


class ValidationSet(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
    root/scene_1/0000000.jpg
    root/scene_1/0000000.npy
    root/scene_1/0000001.jpg
    root/scene_1/0000001.npy
    ..
    root/scene_2/0000000.jpg
    root/scene_2/0000000.npy
    .

    transform functions must take in a list a images and a numpy array which can be None
    """

    def __init__(self, root, transform=None, dataset="nyu"):
        self.root = Path(root)
        scene_list_path = self.root / "val.txt"
        self.scenes = [self.root / folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.dataset = dataset
        self.imgs, self.depth = crawl_folders(self.scenes, self.dataset)

    def __getitem__(self, index):
        img = imread(self.imgs[index]).astype(np.float32)

        if self.dataset == "nyu":
            depth = torch.from_numpy(imread(self.depth[index]).astype(np.float32)).float() / 5000
        elif self.dataset == "kitti":
            depth = torch.from_numpy(np.load(self.depth[index]).astype(np.float32))

        if self.transform is not None:
            img, _ = self.transform([img], None)
            img = img[0]

        return img, depth

    def __len__(self):
        return len(self.imgs)
