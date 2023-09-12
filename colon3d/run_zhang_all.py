import argparse
from pathlib import Path

from colon3d.run_on_sim_dataset import SlamOnDatasetRunner
from colon3d.sim_import.import_sim_dataset import SimImporter
from colon3d.util.general_util import ArgsHelpFormatter, bool_arg, save_unified_results_table

# --------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(formatter_class=ArgsHelpFormatter)

parser.add_argument(
    "--raw_dataset_path",
    type=str,
    help="The path to the dataset to run the algorithm on",
    default="data_gcp/raw_datasets/Zhang22",
)
parser.add_argument(
    "--processed_dataset_path",
    type=str,
    help="The path to save the processed dataset",
    default="data_gcp/datasets/Zhang22",
)
parser.add_argument(
    "--results_base_path",
    type=str,
    default="data/results/Zhang22_v2",
    help="Base path for the results",
)
parser.add_argument(
    "--overwrite_data",
    type=bool_arg,
    default=True,
    help="If True then the pre-processed data folders will be overwritten if they already exists",
)
parser.add_argument(
    "--overwrite_results",
    type=bool_arg,
    default=True,
    help="If True then the save folders will be overwritten if they already exists",
)
parser.add_argument(
    "--debug_mode",
    type=bool_arg,
    help="If true, only one scene will be processed, results will be saved to a debug folder",
    default=False,
)

args = parser.parse_args()
print(f"args={args}")

rand_seed = 0  # random seed for reproducibility
# path to the raw data generate by the unity simulator:
raw_sim_data_path = Path(args.raw_dataset_path)
# path to save the processed scenes dataset:
scenes_dataset_path = Path( args.processed_dataset_path)

alg_settings_override_common = {}

# base path to save the algorithm runs results:
base_results_path = Path(args.results_base_path)
# --------------------------------------------------------------------------------------------------------------------

if args.debug_mode:
    limit_n_scenes = 1  # num scenes to import
    limit_n_frames = 100  # num frames to import from each scene
    n_cases_per_scene = 1  # num cases to generate from each scene
    scenes_dataset_path = scenes_dataset_path.parent / ("_debug_" + scenes_dataset_path.name)
    base_results_path = base_results_path.parent / ("_debug_" + base_results_path.name)
    n_cases_lim = 1  # num cases to run the algorithm on
else:
    limit_n_scenes = 0  # 0 means no limit
    limit_n_frames = 0  # 0 means no limit
    n_cases_per_scene = 5  # num cases to generate from each scene
    n_cases_lim = 0  # 0 means no limit

# --------------------------------------------------------------------------------------------------------------------

# Importing a raw dataset of scenes from the unity simulator:
SimImporter(
    load_path=raw_sim_data_path,
    processed_sim_data_path=scenes_dataset_path,
    limit_n_scenes=limit_n_scenes,
    limit_n_frames=limit_n_frames,
    save_overwrite=args.overwrite_data,
    sim_name="Zhang22",
).run()

# --------------------------------------------------------------------------------------------------------------------
# Run the SLAM algorithm on a dataset of simulated examples:
# --------------------------------------------------------------------------------------------------------------------

common_args = {
    "save_raw_outputs": False,
    "alg_fov_ratio": 0,
    "n_frames_lim": 0,
    "n_scenes_lim": n_cases_lim,
    "save_overwrite": args.overwrite_results,
    "plot_aided_nav": False,
}
# --------------------------------------------------------------------------------------------------------------------
# using the ground truth egomotions - without bundle adjustment
SlamOnDatasetRunner(
    dataset_path=scenes_dataset_path,
    save_path=base_results_path / "no_BA_with_GT_ego",
    depth_maps_source="none",
    egomotions_source="ground_truth",
    alg_settings_override={"use_bundle_adjustment": False} | alg_settings_override_common,
    **common_args,
).run()
save_unified_results_table(base_results_path)
# --------------------------------------------------------------------------------------------------------------------

# Bundle-adjustment, without monocular depth and egomotion estimation
SlamOnDatasetRunner(
    dataset_path=scenes_dataset_path,
    save_path=base_results_path / "BA_no_depth_no_ego",
    depth_maps_source="none",
    egomotions_source="none",
    alg_settings_override=alg_settings_override_common,
    **common_args,
).run()
save_unified_results_table(base_results_path)
# --------------------------------------------------------------------------------------------------------------------
# Bundle-adjustment, with the original EndoSFM monocular depth and egomotion estimation
SlamOnDatasetRunner(
    dataset_path=scenes_dataset_path,
    save_path=base_results_path / "BA_with_EndoSFM_orig",
    depth_maps_source="online_estimates",
    egomotions_source="online_estimates",
    depth_and_egomotion_method="EndoSFM",
    depth_and_egomotion_model_path="saved_models/EndoSFM_orig",
    alg_settings_override=alg_settings_override_common,
    **common_args,
).run()
save_unified_results_table(base_results_path)
# --------------------------------------------------------------------------------------------------------------------
# the original EndoSFM monocular depth and egomotion estimation, with no bundle adjustment
SlamOnDatasetRunner(
    dataset_path=scenes_dataset_path,
    save_path=base_results_path / "no_BA_with_EndoSFM_orig",
    depth_maps_source="online_estimates",
    egomotions_source="online_estimates",
    depth_and_egomotion_method="EndoSFM",
    depth_and_egomotion_model_path="saved_models/EndoSFM_orig",
    alg_settings_override={"use_bundle_adjustment": False} | alg_settings_override_common,
    **common_args,
).run()
save_unified_results_table(base_results_path)

# --------------------------------------------------------------------------------------------------------------------
