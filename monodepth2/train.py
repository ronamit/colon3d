# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
import argparse

from monodepth2.options import MonoDepth2Options, StereoDepthOptions
from monodepth2.trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default="StereoDepth", choices=["MonoDepth2", "StereoDepth"])

# --------------------------------------------------------------------------------------------------------------------

def main():
    args = parser.parse_args()
    if args.method == "MonoDepth2":
        options = MonoDepth2Options()
    elif args.method == "StereoDepth":
        options = StereoDepthOptions()
    else:
        raise NotImplementedError(f"Method {args.method} not implemented")
    opts = options.parse()
    trainer = Trainer(opts)
    trainer.train()
# --------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
# --------------------------------------------------------------------------------------------------------------------
