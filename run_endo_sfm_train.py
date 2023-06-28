from pathlib import Path

from colon3d.sim_import.sim_importer import SimImporter
from endo_sfm.train import TrainRunner

# --------------------------------------------------------------------------------------------------------------------
# Set this True for a new run, False to continue previous runs:

save_overwrite = True  # if True, existing files on the save paths will be overwritten
# if False, the run  will skip functions runs for save paths are not empty

# --------------------------------------------------------------------------------------------------------------------
rand_seed = 1 # random seed for reproducibility
train_dataset_name = "TrainData22"
# path to the raw data generate by the unity simulator:
raw_sim_data_path = Path(f"data/raw_sim_data/{train_dataset_name}")

# path to save the processed scenes dataset:
train_scenes_dataset_path = Path(f"data/sim_data/{train_dataset_name}")

# path to save the trained model:
path_to_save_model = Path("saved_models/EndoSFM_tuned")

# --------------------------------------------------------------------------------------------------------------------

# Importing a raw dataset of scenes from the unity simulator:
sim_importer = SimImporter(
    raw_sim_data_path=raw_sim_data_path,
    processed_sim_data_path=train_scenes_dataset_path,
    save_overwrite=save_overwrite,
)
sim_importer.run()

# --------------------------------------------------------------------------------------------------------------------


train_runner = TrainRunner(save_path_path=path_to_save_model,
                           dataset_path=train_scenes_dataset_path,
                           save_overwrite=save_overwrite)
train_runner.run()

# --------------------------------------------------------------------------------------------------------------------
