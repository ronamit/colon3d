import argparse
from pathlib import Path

import pandas as pd

from colon3d.run_on_sim_dataset import SlamOnDatasetRunner
from colon3d.sim_import.create_target_cases import CasesCreator
from colon3d.sim_import.sim_importer import SimImporter
from colon3d.util.general_util import ArgsHelpFormatter, bool_arg, create_empty_folder

# --------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(formatter_class=ArgsHelpFormatter)

parser.add_argument(
    "--test_dataset_name",
    type=str,
    help="The name of the dataset to run the algorithm on",
    default="TestData21",  # "TestData21" | "SanityCheck23"
)

parser.add_argument(
    "--results_name",
    type=str,
    help="The name of the results folder",
    default="TestData21_results",
)
parser.add_argument(
    "--overwrite_data",
    type=bool_arg,
    default=False,
    help="If True then the pre-processed data folders will be overwritten if they already exists",
)
parser.add_argument(
    "--overwrite_results",
    type=bool_arg,
    default=False,
    help="If True then the results folders will be overwritten if they already exists",
)
parser.add_argument(
    "--debug_mode",
    type=bool_arg,
    help="If true, only one scene will be processed",
    default=False,
)

parser.add_argument(
    "--sanity_check_mode",
    type=bool_arg,
    help="If true, we generate easy cases for sanity check",
    default=False,  # "False"
)
args = parser.parse_args()
print(f"args={args}")
overwrite_results = args.overwrite_results
overwrite_data = args.overwrite_data
debug_mode = args.debug_mode

# --------------------------------------------------------------------------------------------------------------------
rand_seed = 0  # random seed for reproducibility
test_dataset_name = args.test_dataset_name
# path to the raw data generate by the unity simulator:
raw_sim_data_path = Path(f"data/raw_sim_data/{test_dataset_name}")

# path to save the processed scenes dataset:
scenes_dataset_path = Path(f"data/sim_data/{test_dataset_name}")

# path to save the dataset of cases with randomly generated targets added to the original scenes:
scenes_cases_dataset_path = Path(f"data/sim_data/{test_dataset_name}_cases")

# base path to save the algorithm runs results:
base_results_path = Path("results") / args.results_name


if debug_mode:
    limit_n_scenes = 1  # num scenes to import
    limit_n_frames = 100  # num frames to import from each scene
    n_cases_per_scene = 1  # num cases to generate from each scene
    scenes_dataset_path = scenes_dataset_path / "debug"
    scenes_cases_dataset_path = scenes_cases_dataset_path / "debug"
    base_results_path = base_results_path / "debug"
    n_cases_lim = 1  # num cases to run the algorithm on
else:
    limit_n_scenes = 0  # 0 means no limit
    limit_n_frames = 0  # 0 means no limit
    n_cases_per_scene = 5  # num cases to generate from each scene
    n_cases_lim = 0  # 0 means no limit

# in sanity check mode we generate easy cases for sanity check (the target may always be visible)
min_non_visible_frames = 0 if args.sanity_check_mode else 20
# --------------------------------------------------------------------------------------------------------------------

if overwrite_results:
    create_empty_folder(base_results_path)
# --------------------------------------------------------------------------------------------------------------------

# Importing a raw dataset of scenes from the unity simulator:
SimImporter(
    raw_sim_data_path=raw_sim_data_path,
    processed_sim_data_path=scenes_dataset_path,
    limit_n_scenes=limit_n_scenes,
    limit_n_frames=limit_n_frames,
    save_overwrite=overwrite_data,
).run()

# --------------------------------------------------------------------------------------------------------------------

# Generate several cases from each scene, each with randomly chosen target location and size.
CasesCreator(
    sim_data_path=scenes_dataset_path,
    path_to_save_cases=scenes_cases_dataset_path,
    n_cases_per_scene=n_cases_per_scene,
    min_non_visible_frames=min_non_visible_frames,
    rand_seed=rand_seed,
    save_overwrite=overwrite_data,
).run()

# --------------------------------------------------------------------------------------------------------------------
# Run the algorithm on a dataset of simulated examples:
# --------------------------------------------------------------------------------------------------------------------

common_args = {
    "save_raw_outputs": False,
    "alg_fov_ratio": 0,
    "n_frames_lim": 0,
    "n_scenes_lim": n_cases_lim,
    "save_overwrite": overwrite_results,
}
# # --------------------------------------------------------------------------------------------------------------------
# # using the ground truth depth maps and egomotions - without bundle adjustment
# SlamOnDatasetRunner(
#     dataset_path=scenes_cases_dataset_path,
#     save_path=base_results_path / "no_BA_with_GT_depth_and_ego",
#     depth_maps_source="ground_truth",
#     egomotions_source="ground_truth",
#     alg_settings_override={"use_bundle_adjustment": False},
#     **common_args,
# ).run()
# --------------------------------------------------------------------------------------------------------------------

# Bundle-adjustment, without monocular depth and egomotion estimation
SlamOnDatasetRunner(
    dataset_path=scenes_cases_dataset_path,
    save_path=base_results_path / "BA_no_depth_no_ego",
    depth_maps_source="none",
    egomotions_source="none",
    **common_args,
).run()
# --------------------------------------------------------------------------------------------------------------------
# Bundle-adjustment, with the original EndoSFM monocular depth and egomotion estimation
SlamOnDatasetRunner(
    dataset_path=scenes_cases_dataset_path,
    save_path=base_results_path / "BA_with_EndoSFM_orig",
    depth_maps_source="online_estimates",
    egomotions_source="online_estimates",
    depth_and_egomotion_method="EndoSFM",
    depth_and_egomotion_model_path="saved_models/EndoSFM_orig",
    **common_args,
).run()

# --------------------------------------------------------------------------------------------------------------------
# the original EndoSFM monocular depth and egomotion estimation, with no bundle adjustment
SlamOnDatasetRunner(
    dataset_path=scenes_cases_dataset_path,
    save_path=base_results_path / "no_BA_with_EndoSFM_orig",
    depth_maps_source="online_estimates",
    egomotions_source="online_estimates",
    depth_and_egomotion_method="EndoSFM",
    depth_and_egomotion_model_path="saved_models/EndoSFM_orig",
    alg_settings_override={"use_bundle_adjustment": False},
    **common_args,
).run()

# --------------------------------------------------------------------------------------------------------------------
# # Bundle-adjustment, using the ground truth depth maps no egomotions
# SlamOnDatasetRunner(
#     dataset_path=scenes_cases_dataset_path,
#     save_path=base_results_path / "BA_with_GT_depth_no_ego",
#     depth_maps_source="ground_truth",
#     egomotions_source="none",
#     **common_args,
# ).run()
# # --------------------------------------------------------------------------------------------------------------------
# # Bundle-adjustment, using the ground truth depth maps and egomotions.
# SlamOnDatasetRunner(
#     dataset_path=scenes_cases_dataset_path,
#     save_path=base_results_path / "BA_with_GT_depth_and_ego",
#     depth_maps_source="ground_truth",
#     egomotions_source="ground_truth",
#     **common_args,
# ).run()
# --------------------------------------------------------------------------------------------------------------------

# save unified results table for all the runs:
unified_results_table = pd.DataFrame()
for results_path in base_results_path.glob("*"):
    if results_path.is_dir():
        cur_result_path = results_path / "metrics_summary.csv"
        if not cur_result_path.exists():
            continue
        # load the current run results summary csv file:
        run_results_summary = pd.read_csv(cur_result_path)
        # add the run name to the results table:
        unified_results_table = pd.concat([unified_results_table, run_results_summary], axis=0)
# save the unified results table:
file_path = base_results_path / "unified_results_table.csv"
unified_results_table.to_csv(file_path, encoding="utf-8", index=False)
print(f"Saved unified results table to {file_path}")

# --------------------------------------------------------------------------------------------------------------------
