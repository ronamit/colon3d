import argparse
from pathlib import Path

from colon_nav.run_on_sim_dataset import SlamOnDatasetRunner
from colon_nav.util.general_util import ArgsHelpFormatter, bool_arg, save_unified_results_table

# --------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(formatter_class=ArgsHelpFormatter)

parser.add_argument(
    "--dataset_path",
    type=str,
    help="Path to a dataset of scenes.",
    default="data_gcp/datasets/Zhang22",
)
parser.add_argument(
    "--results_base_path",
    type=str,
    default="data/results/Zhang22",
    help="Base path for the results",
)
parser.add_argument(
    "--models_base_path",
    type=str,
    default="data_gcp/models",
)
parser.add_argument(
    "--save_overwrite",
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
# path to save the processed scenes dataset:
dataset_path = Path(args.dataset_path)
models_base_path = Path(args.models_base_path)
alg_settings_override_common = {}

# base path to save the algorithm runs results:
base_results_path = Path(args.results_base_path)
# --------------------------------------------------------------------------------------------------------------------

if args.debug_mode:
    print("Running in debug mode!!!!")
    limit_n_scenes = 1  # num scenes to run on (0 means no limit)
    limit_n_frames = 10  # num frames to run on (0 means no limit)
    base_results_path = base_results_path / "debug"
    print_interval = 1  # print progress every frame
else:
    limit_n_scenes = 0  # 0 means no limit]
    limit_n_frames = 0  # 0 means no limit
    print_interval = 20  # print progress every X frames


# --------------------------------------------------------------------------------------------------------------------
# Run the SLAM algorithm on a dataset of simulated examples:
# --------------------------------------------------------------------------------------------------------------------

common_args = {
    "save_raw_outputs": False,
    "alg_fov_ratio": 0,
    "n_frames_lim": limit_n_frames,
    "n_scenes_lim": limit_n_scenes,
    "save_overwrite": args.save_overwrite,
    "load_scenes_with_targets": False,  # The Zhang22 dataset does not have targets
    "print_interval": print_interval,
}

# --------------------------------------------------------------------------------------------------------------------

# Bundle-adjustment, without monocular depth and egomotion estimation
SlamOnDatasetRunner(
    dataset_path=dataset_path,
    save_path=base_results_path / "BA_no_depth_no_ego",
    depth_maps_source="none",
    egomotions_source="none",
    alg_settings_override=alg_settings_override_common,
    **common_args,
).run()
save_unified_results_table(base_results_path)

# --------------------------------------------------------------------------------------------------------------------
# using the ground truth egomotions - without bundle adjustment
SlamOnDatasetRunner(
    dataset_path=dataset_path,
    save_path=base_results_path / "no_BA_with_GT_ego",
    depth_maps_source="none",
    egomotions_source="ground_truth",
    alg_settings_override={"use_bundle_adjustment": False} | alg_settings_override_common,
    **common_args,
).run()
save_unified_results_table(base_results_path)

# --------------------------------------------------------------------------------------------------------------------
# Bundle-adjustment, with the original EndoSFM monocular depth and egomotion estimation
SlamOnDatasetRunner(
    dataset_path=dataset_path,
    save_path=base_results_path / "BA_with_EndoSFM_orig",
    depth_maps_source="online_estimates",
    egomotions_source="online_estimates",
    depth_and_egomotion_method="EndoSFM",
    model_path=models_base_path / "EndoSFM_orig",
    alg_settings_override=alg_settings_override_common,
    **common_args,
).run()
save_unified_results_table(base_results_path)
# --------------------------------------------------------------------------------------------------------------------
# the original EndoSFM monocular depth and egomotion estimation, with no bundle adjustment
SlamOnDatasetRunner(
    dataset_path=dataset_path,
    save_path=base_results_path / "no_BA_with_EndoSFM_orig",
    depth_maps_source="online_estimates",
    egomotions_source="online_estimates",
    depth_and_egomotion_method="EndoSFM",
    model_path=models_base_path / "EndoSFM_orig",
    alg_settings_override={"use_bundle_adjustment": False} | alg_settings_override_common,
    **common_args,
).run()
save_unified_results_table(base_results_path)

# --------------------------------------------------------------------------------------------------------------------
