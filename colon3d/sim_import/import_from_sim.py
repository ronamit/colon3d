import argparse

from colon3d.sim_import.sim_importer import SimImporter
from colon3d.utils.general_util import ArgsHelpFormatter

# --------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(formatter_class=ArgsHelpFormatter)
    parser.add_argument(
        "--raw_sim_data_path",
        type=str,
        default="data/raw_sim_data/SimData14_test",
        help="The path to the dir with the captured scenes generated by the Unity simulator",
    )
    parser.add_argument(
        "--processed_sim_data_path",
        type=str,
        default="data/sim_data/SimData14_test",
        help="The path to save the processed simulated scene",
    )
    parser.add_argument(
        "--ask_overwrite",
        type=bool,
        default=False,
        help="If True, the user will be asked to confirm overwriting existing files",
    )
    parser.add_argument(
        "--limit_n_scenes",
        type=int,
        default=0,
        help="The number maximal number of scenes to process, if 0 all scenes will be processed",
    )
    parser.add_argument(
        "--limit_n_frames",
        type=int,
        default=0,
        help="The number maximal number of frames to take from each scenes, if 0 all frames will be processed",
    )
    parser.add_argument(
        "--fps_override",
        type=float,
        default=0,
        help="frame rate in Hz of the output videos, if 0 the frame rate will be extracted from the settings file",
    )
    args = parser.parse_args()

    sim_importer = SimImporter(
        raw_sim_data_path=args.raw_sim_data_path,
        processed_sim_data_path=args.processed_sim_data_path,
        limit_n_scenes=args.limit_n_scenes,
        limit_n_frames=args.limit_n_frames,
        fps_override=args.fps_override,
        ask_overwrite=args.ask_overwrite,
    )
    sim_importer.import_data()


# --------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
