import argparse
from pathlib import Path

from colon3d.sim_import.import_dataset import SimImporter
from colon3d.util.general_util import ArgsHelpFormatter, Tee, bool_arg

# -------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(formatter_class=ArgsHelpFormatter)
    parser.add_argument(
        "--load_path",
        type=str,
        default="data_gcp/raw_datasets/SimCol3D",
        help="Path to load raw dataset ",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="The path to save the prepared dataset",
        default="data_gcp/datasets/SimCol3D",
    )
    parser.add_argument(
        "--save_overwrite",
        type=bool_arg,
        default=True,
        help="If True then scenes folders will be overwritten if they already exists",
    )
    parser.add_argument(
        "--debug_mode",
        type=bool_arg,
        help="If true, only one scene will be processed",
        default=False,
    )
    args = parser.parse_args()
    save_overwrite = args.save_overwrite
    debug_mode = args.debug_mode

    # --------------------------------------------------------------------------------------------------------------------

    limit_n_scenes = 0  # no limit
    limit_n_frames = 0  #  no limit

    if debug_mode:
        limit_n_scenes = 1  # num scenes to import
        limit_n_frames = 50

    # --------------------------------------------------------------------------------------------------------------------
    # path to the raw data generate by the unity simulator:
    load_path = Path(args.load_path)

    # path to save the processed scenes dataset:
    save_path = Path(args.save_path)

    # --------------------------------------------------------------------------------------------------------------------

    with Tee(save_path / "log.txt"):  # save the prints to a file
        for split_name in ["Train", "Test"]:
            # Importing a raw dataset of scenes from the unity simulator:
            SimImporter(
                load_path=load_path,
                split_name=split_name,
                save_path=save_path / split_name,
                limit_n_scenes=limit_n_scenes,
                limit_n_frames=limit_n_frames,
                save_overwrite=save_overwrite,
                sim_name="SimCol3D",
            ).run()


# --------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------------------------------------------------
