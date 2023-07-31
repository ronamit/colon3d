import argparse
from pathlib import Path

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
    "--sanity_check_mode",
    type=bool_arg,
    help="If true, we generate easy cases for sanity check",
    default=False,  # "False"
)
args = parser.parse_args()
save_overwrite = args.save_overwrite
debug_mode = args.debug_mode
print(
    f"test_dataset_name={args.test_dataset_name}, save_overwrite={save_overwrite}, debug_mode={debug_mode}, sanity_check_mode={args.sanity_check_mode}",
)

# --------------------------------------------------------------------------------------------------------------------
rand_seed = 0  # random seed for reproducibility
test_dataset_name = args.test_dataset_name
# path to the raw data generate by the unity simulator:
raw_sim_data_path = Path(f"data/raw_sim_data/{test_dataset_name}")

# path to save the processed scenes dataset:
scenes_dataset_path = Path(f"data/sim_data/{test_dataset_name}")

# base path to save the algorithm runs results:
base_results_path = Path(f"results/{test_dataset_name}_results")


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

# in sanity check mode we generate easy cases for sanity check (the target may always be visible)
min_non_visible_frames = 0 if args.sanity_check_mode else 20

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
    "n_cases_lim": n_cases_lim,
    "save_overwrite": save_overwrite,
}
# --------------------------------------------------------------------------------------------------------------------
# using the ground truth egomotions - without bundle adjustment
SlamOnDatasetRunner(
    dataset_path=scenes_dataset_path,
    save_path=base_results_path / "no_BA_with_GT_depth_and_ego",
    depth_maps_source="none",
    egomotions_source="ground_truth",
    use_bundle_adjustment=False,
    **common_args,
).run()
# --------------------------------------------------------------------------------------------------------------------
