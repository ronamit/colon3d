import argparse
from pathlib import Path

from colon3d.general_util import ArgsHelpFormatter

# ---------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(formatter_class=ArgsHelpFormatter)
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/sim_data/SimData8",
        help="Path to the dataset of scenes.",
    )
    args = parser.parse_args()
    dataset_path = Path(args.dataset_path).expanduser()
    
    # plot examples of depth maps (ground truth and estimated)
    
    
    # run depth estimator on all scenes, to get the depths histograms
    
    # get the ground truth depths histograms
    
    
