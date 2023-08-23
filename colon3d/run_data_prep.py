import argparse
from pathlib import Path

from colon3d.sim_import.create_target_cases import TargetCasesCreator
from colon3d.sim_import.sim_importer import SimImporter
from colon3d.util.general_util import ArgsHelpFormatter, Tee, bool_arg

# -------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(formatter_class=ArgsHelpFormatter)
    parser.add_argument(
        "--test_dataset_name",
        type=str,
        help="The name of the dataset to run the algorithm on",
        default="TestData21",  # "TestData21"
    )
    parser.add_argument(
        "--data_base_path",
        type=str,
        default="/mnt/disk1/data",
        help="Base path for the data",
    )
    parser.add_argument(
        "--overwrite_data",
        type=bool_arg,
        default=True,
        help="If True then the pre-processed data folders will be overwritten if they already exists",
    )
    parser.add_argument(
        "--debug_mode",
        type=bool_arg,
        help="If true, only one scene will be processed",
        default=True,
    )
    parser.add_argument(
        "--easy_cases_mode",
        type=bool_arg,
        help="If true, we generate easy cases where the target is always visible",
        default=False,
    )
    args = parser.parse_args()
    overwrite_data = args.overwrite_data
    debug_mode = args.debug_mode
    test_dataset_name = args.test_dataset_name
    processed_dataset_name = test_dataset_name

    # --------------------------------------------------------------------------------------------------------------------

    limit_n_scenes = 0  # no limit
    limit_n_frames = 0  #  no limit
    n_cases_per_scene = 5  # num cases to generate from each scene

    if debug_mode:
        limit_n_scenes = 1  # num scenes to import
        limit_n_frames = 100  # num frames to import from each scene (note - use at least 100 so it will be possible to get a track that goes out of view)
        n_cases_per_scene = 1  # num cases to generate from each scene
        processed_dataset_name = "_debug_" + processed_dataset_name

    # --------------------------------------------------------------------------------------------------------------------
    rand_seed = 0  # random seed for reproducibility

    data_base_path = Path(args.data_base_path)

    # path to the raw data generate by the unity simulator:
    raw_sim_data_path = data_base_path / "raw_sim_data" / test_dataset_name

    # path to save the processed scenes dataset:
    scenes_dataset_path = data_base_path / "sim_data" / processed_dataset_name

    # in sanity check mode we generate easy cases for sanity check (the target may always be visible)
    min_non_visible_frames = 0 if args.easy_cases_mode else 20

    # --------------------------------------------------------------------------------------------------------------------

    with Tee(scenes_dataset_path / "log.txt"):  # save the prints to a file
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
        TargetCasesCreator(
            sim_data_path=scenes_dataset_path,
            n_cases_per_scene=n_cases_per_scene,
            min_non_visible_frames=min_non_visible_frames,
            rand_seed=rand_seed,
            save_overwrite=overwrite_data,
        ).run()


# --------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------------------------------------------------
