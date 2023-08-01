import argparse
from pathlib import Path

import pandas as pd

from colon3d.run_on_sim_dataset import SlamOnDatasetRunner
from colon3d.sim_import.sim_importer import SimImporter
from colon3d.util.general_util import ArgsHelpFormatter, bool_arg

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
    help="If true, only one scene will be processed",
    default=False,
)
parser.add_argument(
    "--test_dataset_name",
    type=str,
    help="The name of the dataset to run the algorithm on",
    default="Zhang22",
)

parser.add_argument(
    "--no_penalties_mode",
    type=bool_arg,
    help="If true, run the algorithm without penalties",
    default=False,
)

args = parser.parse_args()
print(f"args={args}")
save_overwrite = args.save_overwrite
debug_mode = args.debug_mode

# --------------------------------------------------------------------------------------------------------------------
rand_seed = 0  # random seed for reproducibility
test_dataset_name = args.test_dataset_name
# path to the raw data generate by the unity simulator:
raw_sim_data_path = Path(f"data/raw_sim_data/{test_dataset_name}")

# path to save the processed scenes dataset:
scenes_dataset_path = Path(f"data/sim_data/{test_dataset_name}")

alg_settings_override_common = {}
results_name = test_dataset_name

if args.no_penalties_mode:
    alg_settings_override_common = {"add_penalties": False}
    results_name = results_name + "_no_penalties"
    
    # base path to save the algorithm runs results:
base_results_path = Path("results") / results_name


if debug_mode:
    limit_n_scenes = 1  # num scenes to import
    limit_n_frames = 100  # num frames to import from each scene
    n_cases_per_scene = 1  # num cases to generate from each scene
    scenes_dataset_path = scenes_dataset_path / "debug"
    base_results_path = base_results_path / "debug"
    n_cases_lim = 1  # num cases to run the algorithm on
else:
    limit_n_scenes = 0  # 0 means no limit
    limit_n_frames = 0  # 0 means no limit
    n_cases_per_scene = 5  # num cases to generate from each scene
    n_cases_lim = 0  # 0 means no limit

# --------------------------------------------------------------------------------------------------------------------

# Importing a raw dataset of scenes from the unity simulator:
SimImporter(
    raw_sim_data_path=raw_sim_data_path,
    processed_sim_data_path=scenes_dataset_path,
    limit_n_scenes=limit_n_scenes,
    limit_n_frames=limit_n_frames,
    save_overwrite=save_overwrite,
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
    "save_overwrite": save_overwrite,
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
# --------------------------------------------------------------------------------------------------------------------
# Bundle-adjustment, with the original EndoSFM monocular depth and egomotion estimation
SlamOnDatasetRunner(
    dataset_path=scenes_dataset_path,
    save_path=base_results_path / "BA_with_EndoSFM_orig",
    depth_maps_source="online_estimates",
    egomotions_source="online_estimates",
    depth_and_egomotion_model_path="saved_models/EndoSFM_orig",
    alg_settings_override=alg_settings_override_common,
    **common_args,
).run()
# --------------------------------------------------------------------------------------------------------------------
# the original EndoSFM monocular depth and egomotion estimation, with no bundle adjustment
SlamOnDatasetRunner(
    dataset_path=scenes_dataset_path,
    save_path=base_results_path / "no_BA_with_EndoSFM_orig",
    depth_maps_source="online_estimates",
    egomotions_source="online_estimates",
    depth_and_egomotion_model_path="saved_models/EndoSFM_orig",
    alg_settings_override={"use_bundle_adjustment": False} | alg_settings_override_common,
    **common_args,
).run()
# --------------------------------------------------------------------------------------------------------------------

# save unified results table for all the runs:
unified_results_table = pd.DataFrame()
for results_path in base_results_path.glob("*"):
    if results_path.is_dir():
        # load the current run results summary csv file:
        run_results_summary = pd.read_csv(results_path / "metrics_summary.csv")
        # add the run name to the results table:
        unified_results_table = pd.concat([unified_results_table, run_results_summary], axis=0)
# save the unified results table:
unified_results_table.to_csv(base_results_path / "unified_results_table.csv", encoding="utf-8", index=False)
# --------------------------------------------------------------------------------------------------------------------
