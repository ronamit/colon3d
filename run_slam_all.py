import argparse
from pathlib import Path

from colon3d.run_slam_on_sim_dataset import SlamOnDatasetRunner
from colon3d.sim_import.create_cases import CasesCreator
from colon3d.sim_import.sim_importer import SimImporter
from colon3d.utils.general_util import ArgsHelpFormatter, bool_arg

# --------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(formatter_class=ArgsHelpFormatter)
parser.add_argument(
    "--save_overwrite",
    type=bool_arg,
    default=False,
    help="If True then the save folders will be overwritten if they already exists",
)
parser.add_argument(
    "--local_mode" ,
    type=bool_arg,
    help="If true, ray will run in local mode (single process) - useful for debugging",
    default=False,
)
args = parser.parse_args()
save_overwrite = args.save_overwrite
local_mode = args.local_mode
print(f"save_overwrite={save_overwrite}, local_mode={local_mode}")

# --------------------------------------------------------------------------------------------------------------------
rand_seed = 1  # random seed for reproducibility
test_dataset_name = "TestData21"
# path to the raw data generate by the unity simulator:
raw_sim_data_path = Path(f"data/raw_sim_data/{test_dataset_name}")

# path to save the processed scenes dataset:
scenes_dataset_path = Path(f"data/sim_data/{test_dataset_name}")

# path to save the dataset of cases generated from the scenes:
cases_dataset_path = Path(f"data/sim_data/{test_dataset_name}_cases")

# base path to save the algorithm runs results:
base_results_path = Path(f"results/{test_dataset_name}_results")

# --------------------------------------------------------------------------------------------------------------------

# Importing a raw dataset of scenes from the unity simulator:
SimImporter(
    raw_sim_data_path=raw_sim_data_path,
    processed_sim_data_path=scenes_dataset_path,
    save_overwrite=save_overwrite,
).run()

# --------------------------------------------------------------------------------------------------------------------

# Generate several cases from each scene, each with randomly chosen target location and size.
CasesCreator(
    sim_data_path=scenes_dataset_path,
    path_to_save_cases=cases_dataset_path,
    n_cases_per_scene=5,
    rand_seed=rand_seed,
    save_overwrite=save_overwrite,
    local_mode=local_mode,
).run()

# --------------------------------------------------------------------------------------------------------------------
# Run the algorithm on a dataset of simulated examples:
# --------------------------------------------------------------------------------------------------------------------
# using the ground truth depth maps and egomotions - without bundle adjustment
SlamOnDatasetRunner(
    dataset_path=cases_dataset_path,
    save_path=base_results_path / "no_BA_with_GT_depth_and_ego",
    depth_maps_source="ground_truth",
    egomotions_source="ground_truth",
    use_bundle_adjustment=False,
    save_overwrite=save_overwrite,
    local_mode=local_mode,
).run()
# --------------------------------------------------------------------------------------------------------------------

# without monocular depth and egomotion estimation
SlamOnDatasetRunner(
    dataset_path=cases_dataset_path,
    save_path=base_results_path / "BA_no_depth_no_ego",
    depth_maps_source="none",
    egomotions_source="none",
    save_overwrite=save_overwrite,
    local_mode=local_mode,
).run()
# --------------------------------------------------------------------------------------------------------------------
# with the original EndoSFM monocular depth and egomotion estimation
SlamOnDatasetRunner(
    dataset_path=cases_dataset_path,
    save_path=base_results_path / "BA_with_EndoSFM_orig",
    depth_maps_source="online_estimates",
    egomotions_source="online_estimates",
    depth_and_egomotion_model_path="saved_models/EndoSFM_orig",
    save_overwrite=save_overwrite,
    local_mode=local_mode,
).run()
# --------------------------------------------------------------------------------------------------------------------
# with the tuned EndoSFM monocular depth and egomotion estimation
SlamOnDatasetRunner(
    dataset_path=cases_dataset_path,
    save_path=base_results_path / "BA_with_EndoSFM_tuned",
    depth_maps_source="online_estimates",
    egomotions_source="online_estimates",
    depth_and_egomotion_model_path="saved_models/EndoSFM_tuned",
    save_overwrite=save_overwrite,
    local_mode=local_mode,
).run()
# --------------------------------------------------------------------------------------------------------------------
# the original EndoSFM monocular depth and egomotion estimation, with no bundle adjustment
SlamOnDatasetRunner(
    dataset_path=cases_dataset_path,
    save_path=base_results_path / "no_BA_with_EndoSFM_orig",
    depth_maps_source="online_estimates",
    egomotions_source="online_estimates",
    depth_and_egomotion_model_path="saved_models/EndoSFM_orig",
    use_bundle_adjustment=False,
    save_overwrite=save_overwrite,
).run()
# --------------------------------------------------------------------------------------------------------------------
# the tuned EndoSFM monocular depth and egomotion estimation, with no bundle adjustment
SlamOnDatasetRunner(
    dataset_path=cases_dataset_path,
    save_path=base_results_path / "no_BA_with_EndoSFM_tuned",
    depth_maps_source="online_estimates",
    egomotions_source="online_estimates",
    depth_and_egomotion_model_path="saved_models/EndoSFM_tuned",
    use_bundle_adjustment=False,
    save_overwrite=save_overwrite,
    local_mode=local_mode,
).run()
# --------------------------------------------------------------------------------------------------------------------
# using the ground truth depth maps no egomotions
SlamOnDatasetRunner(
    dataset_path=cases_dataset_path,
    save_path=base_results_path / "BA_with_GT_depth_no_ego",
    depth_maps_source="ground_truth",
    egomotions_source="none",
    save_overwrite=save_overwrite,
).run()
# --------------------------------------------------------------------------------------------------------------------
#  using the ground truth depth maps and egomotions - with bundle adjustment
SlamOnDatasetRunner(
    dataset_path=cases_dataset_path,
    save_path=base_results_path / "BA_with_GT_depth_and_ego",
    depth_maps_source="ground_truth",
    egomotions_source="ground_truth",
    save_overwrite=save_overwrite,
    local_mode=local_mode,
).run()
# --------------------------------------------------------------------------------------------------------------------
