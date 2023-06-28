from pathlib import Path

from colon3d.run_slam_on_sim_dataset import SlamOnDatasetRunner
from colon3d.sim_import.create_cases import CasesCreator
from colon3d.sim_import.sim_importer import SimImporter

# --------------------------------------------------------------------------------------------------------------------
# Set this True for a new run, False to continue previous runs:

save_overwrite = False  # if True, existing files on the save paths will be overwritten
# if False, the run  will skip functions runs for save paths are not empty

# --------------------------------------------------------------------------------------------------------------------
rand_seed = 1 # random seed for reproducibility
test_dataset_name = "TestData21"
# path to the raw data generate by the unity simulator:
raw_sim_data_path = Path(f"data/raw_sim_data/{test_dataset_name}")

# path to save the processed scenes dataset:
scenes_dataset_path = Path(f"data/sim_data/{test_dataset_name}")

# path to save the dataset of cases generated from the scenes:
cases_dataset_path = Path(f"data/sim_data/{test_dataset_name}_cases")

# base path to save the algorithm runs results:
base_results_path = Path(f"data/sim_data/{test_dataset_name}_results")

# --------------------------------------------------------------------------------------------------------------------

# Importing a raw dataset of scenes from the unity simulator:
sim_importer = SimImporter(
    raw_sim_data_path=raw_sim_data_path,
    processed_sim_data_path=scenes_dataset_path,
    save_overwrite=save_overwrite,
)
sim_importer.run()

# --------------------------------------------------------------------------------------------------------------------

# Generate several cases from each scene, each with randomly chosen target location and size.
cases_creator = CasesCreator(
    sim_data_path=scenes_dataset_path,
    path_to_save_cases=cases_dataset_path,
    n_cases_per_scene=5,
    rand_seed=rand_seed,
    save_overwrite=save_overwrite,
)
cases_creator.run()

# --------------------------------------------------------------------------------------------------------------------
# Run the algorithm on a dataset of simulated examples:
# --------------------------------------------------------------------------------------------------------------------

# 1) without monocular depth and egomotion estimation
slam_on_dataset_runner = SlamOnDatasetRunner(
    dataset_path=cases_dataset_path,
    save_path=base_results_path / "SLAM_without_monocular_estimation",
    depth_maps_source="none",
    egomotions_source="none",
    save_overwrite=save_overwrite,
)
slam_on_dataset_runner.run()
# --------------------------------------------------------------------------------------------------------------------
# 2) with the original EndoSFM monocular depth and egomotion estimation
slam_on_dataset_runner = SlamOnDatasetRunner(
    dataset_path=cases_dataset_path,
    save_path=base_results_path / "SLAM_with_EndoSFM_orig",
    depth_maps_source="online_estimates",
    egomotions_source="online_estimates",
    depth_and_egomotion_model_path="saved_models/EndoSFM_orig",
    save_overwrite=save_overwrite,
)
slam_on_dataset_runner.run()
# --------------------------------------------------------------------------------------------------------------------
# 3) with the tuned EndoSFM monocular depth and egomotion estimation
slam_on_dataset_runner = SlamOnDatasetRunner(
    dataset_path=cases_dataset_path,
    save_path=base_results_path / "SLAM_with_EndoSFM_tuned",
    depth_maps_source="online_estimates",
    egomotions_source="online_estimates",
    depth_and_egomotion_model_path="saved_models/EndoSFM_tuned",
    save_overwrite=save_overwrite,
)