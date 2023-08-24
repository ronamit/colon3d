import random
from pathlib import Path

from colon3d.util.data_util import get_all_scenes_paths_in_dir

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


# -----------------------------------------F----------------------------------------------------------------------------
