import shutil
from pathlib import Path

from colon_nav.util.general_util import create_empty_folder

# --------------------------------------------------------------------------------------------------------------------

input_dataset_paths = ["data/datasets/SimCol3D/Train", "data/datasets/ColonNav/Train"]

output_dataset_path = "data/datasets/UnifiedTrainSet"

# --------------------------------------------------------------------------------------------------------------------

input_dataset_paths = [Path(input_dataset_path) for input_dataset_path in input_dataset_paths]
output_dataset_path = Path(output_dataset_path)

scenes_path_to_load = []
for input_dataset_path in input_dataset_paths:
    # make sure the input dataset path exists:
    assert Path(input_dataset_path).exists(), f"input dataset path {input_dataset_path} does not exist"
    # get the scenes paths to load:
    cur_scenes_path_to_load = list(Path(input_dataset_path).glob("Scene_*"))
    print(f"found {len(cur_scenes_path_to_load)} scenes in {input_dataset_path}")
    scenes_path_to_load.extend(cur_scenes_path_to_load)

print(f"found {len(scenes_path_to_load)} scenes in total")

create_empty_folder(output_dataset_path, save_overwrite=True)

for i_scene, load_scene_path in enumerate(scenes_path_to_load):
    save_path = Path(output_dataset_path) / f"Scene_{i_scene:05d}"
    # copy the scene folder:
    shutil.copytree(load_scene_path, save_path)
    # rename the scene folder:
    save_path.rename(Path(output_dataset_path) / f"Scene_{i_scene:05d}")
    # print progress:
    print(f"copied scene {i_scene} of {len(scenes_path_to_load)}")

print("done")
# --------------------------------------------------------------------------------------------------------------------
