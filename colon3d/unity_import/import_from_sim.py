import argparse

from colon3d.unity_import.old_sim_importer import OldSimImporter

# --------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_sim_data_path",
        type=str,
        default="data/raw_sim_data/Seq_00009",
        help="The path to the Unity simulator generated data of a single sequence",
    )
    parser.add_argument(
        "--sim_type",
        type=str,
        default="old_sim",
        help="The type of simulator used to generate the data, can be 'old_sim' or 'new_sim'",
    )
        
    parser.add_argument(
        "--path_to_save_sequence",
        type=str,
        default="data/sim_data/Seq_00009_short_TEMP",
        help="The path to save the processed simulated sequence",
    )
    parser.add_argument(
        "--fps_override",
        type=float,
        default=0,
        help="frame rate in Hz of the output videos, if 0 the frame rate will be extracted from the settings file",
    )
    parser.add_argument(
        "--limit_n_frames",
        type=int,
        default=200,
        help="The number of frames to process, if 0 all frames will be processed",
    )

    args = parser.parse_args()
    if args.sim_type == "old_sim":
        sim_importer = OldSimImporter(args.raw_sim_data_path, args.path_to_save_sequence, args.fps_override, args.limit_n_frames)
        sim_importer.import_sequence()
    else:
        raise ValueError(f"Unknown simulator type: {args.sim_type}")


# --------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
