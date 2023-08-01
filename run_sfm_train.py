import argparse
from pathlib import Path

import yaml

from colon3d.examine_depths import DepthExaminer
from colon3d.sim_import.sim_importer import SimImporter
from colon3d.util.general_util import ArgsHelpFormatter, bool_arg, save_dict_to_yaml
from endo_sfm.train import TrainRunner
from endo_sfm.utils import save_model_info

# --------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(formatter_class=ArgsHelpFormatter)
parser.add_argument(
    "--save_overwrite",
    type=bool_arg,
    default=True,
    help="If True then the save folders will be overwritten if they already exists",
)
parser.add_argument(
    "--debug_mode",
    type=bool_arg,
    default=True,
    help="If True, then use a small dataset and 1 epoch for debugging",
)
args = parser.parse_args()
print(f"args={args}")
save_overwrite = args.save_overwrite

# --------------------------------------------------------------------------------------------------------------------
rand_seed = 0  # random seed for reproducibility
train_dataset_name = "TrainData22"
# path to the raw data generate by the unity simulator:
raw_sim_data_path = Path(f"data/raw_sim_data/{train_dataset_name}")

# path to save the processed scenes dataset:
train_scenes_dataset_path = Path(f"data/sim_data/{train_dataset_name}")

# path to save the trained model:
path_to_save_model = Path("saved_models/EndoSFM_tuned")

# path of the pretrained models:
pretrained_disp = "saved_models/EndoSFM_orig/DispNet_best.pt"
pretrained_pose = "saved_models/EndoSFM_orig/PoseNet_best.pt"


path_to_save_depth_exam = path_to_save_model / "depth_exam"

if args.debug_mode:
    limit_n_scenes = 10
    limit_n_frames = 10
    n_epochs = 1
    path_to_save_model = path_to_save_model / "debug"
else:
    limit_n_scenes = 0
    limit_n_frames = 0
    n_epochs = 100

model_description = (
    "Models are defined in https://github.com/CapsuleEndoscope/EndoSLAM."
    "initial weights were downloaded from  https://github.com/CapsuleEndoscope/VirtualCapsuleEndoscopy (best checkpoint),"
    f"Trained for {n_epochs} epochs on  {train_dataset_name}"
)
# --------------------------------------------------------------------------------------------------------------------

# Importing a raw dataset of scenes from the unity simulator:
sim_importer = SimImporter(
    raw_sim_data_path=raw_sim_data_path,
    processed_sim_data_path=train_scenes_dataset_path,
    save_overwrite=save_overwrite,
    limit_n_scenes=limit_n_scenes,
    limit_n_frames=limit_n_frames,
)
scenes_paths = sim_importer.run()
# --------------------------------------------------------------------------------------------------------------------
# Run training:

train_runner = TrainRunner(
    save_path=path_to_save_model,
    dataset_path=train_scenes_dataset_path,
    pretrained_disp=pretrained_disp,
    pretrained_pose=pretrained_pose,
    save_overwrite=save_overwrite,
    n_epochs=n_epochs,
)
train_runner.run()

# --------------------------------------------------------------------------------------------------------------------
# Save model info:

# get scene metata for some example scene in the train dataset (should be the same for all scenes):
scene_path = scenes_paths[0]
with (scene_path / "meta_data.yaml").open("r") as f:
    scene_metadata = yaml.load(f, Loader=yaml.FullLoader)

# set this value initially to 1.0, and then run the depth examination to calibrate the depth scale:
net_out_to_mm = 1.0

save_model_info(
    save_dir_path=path_to_save_model,
    scene_metadata=scene_metadata,
    disp_resnet_layers=train_runner.disp_resnet_layers,
    pose_resnet_layers=train_runner.pose_resnet_layers,
    overwrite=True,
    extra_info={"model_description": model_description, "net_out_to_mm": net_out_to_mm},
)

# --------------------------------------------------------------------------------------------------------------------
# Run depth examination to calibrate the depth scale:
depth_examiner = DepthExaminer(
    dataset_path=train_scenes_dataset_path,
    depth_and_egomotion_model_path=path_to_save_model,
    save_path=path_to_save_depth_exam,
    save_overwrite=save_overwrite,
)
depth_exam = depth_examiner.run()

# the output of the depth network needs to be multiplied by this number to get the depth in mm (based on the analysis of the true depth data in examine_depths.py)
net_out_to_mm = depth_exam["avg_gt_depth / avg_est_depth"]
info_model_path = path_to_save_model / "model_info.yaml"
# update the model info file with the new net_out_to_mm value:
with info_model_path.open("r") as f:
    model_info = yaml.load(f, Loader=yaml.FullLoader)
    model_info["net_out_to_mm"] = net_out_to_mm
save_dict_to_yaml(save_path=info_model_path, dict_to_save=model_info)
print(f"Updated model info file {info_model_path} with the calibrated net_out_to_mm value: {net_out_to_mm} [mm]")

# --------------------------------------------------------------------------------------------------------------------
