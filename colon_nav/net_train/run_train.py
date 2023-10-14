import argparse
import os
from pathlib import Path

import numpy as np
import torch

from colon_nav.examine_depths import DepthExaminer
from colon_nav.net_train.net_trainer import NetTrainer
from colon_nav.net_train.scenes_dataset import ScenesDataset, get_scenes_dataset_random_split
from colon_nav.net_train.train_utils import ModelInfo, save_model_info
from colon_nav.util.general_util import ArgsHelpFormatter, bool_arg, get_path, set_rand_seed

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"  # prevent cuda out of memory error
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

"""
If out-of-memory error occurs, try to reduce the batch size (e.g. --batch_size 4)
"""


# -------------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(formatter_class=ArgsHelpFormatter)
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data_gcp/datasets/UnifiedTrainSet",
        help="Path to the dataset of scenes used for training (not raw data, i.e., output of import_dataset.py ).",
    )
    parser.add_argument(
        "--validation_ratio",
        type=float,
        default=0.1,
        help="ratio of the number of scenes in the validation set from entire training set scenes",
    )
    parser.add_argument(
        "--n_ref_imgs",
        type=int,
        default=2,
        help="Number of reference images. Must be at least 1. If the target is at frame t, then the reference frames are at frames  t - n_ref_imgs, ..., t - 1",
    )
    parser.add_argument(
        "--train_method",
        type=str,
        default="GT_pose_and_depth",
        choices=["GT_pose_and_depth", "GT_depth_only", "self_supervised"],
        help="Method to use for training.",
    )
    parser.add_argument(
        "--egomotion_model_name",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet50"],
        help="Name of the egomotion model.",
    )
    parser.add_argument(
        "--depth_model_name",
        type=str,
        default="DenseDepth",
        choices=["FCBFormer"],
        help="Name of the depth model.",
    )
    parser.add_argument(
        "--load_egomotion_model_path",
        type=str,
        default="",
        help="Path to the pretrained egomotion model. If empty string then use the default ImageNet pretrained weights."
        "Note that a loaded pretrained model must be compatible with the egomotion_model_name and n_ref_imgs.",
    )
    parser.add_argument(
        "--load_depth_model_path",
        type=str,
        default="",
        help="Path to the pretrained depth model. If empty string then use the default ImageNet pretrained weights."
        "Note that a loaded pretrained model must be compatible with the depth_model_name and n_ref_imgs.",
    )
    parser.add_argument(
        "--path_to_save_models",
        type=str,
        default="data_gcp/models/TEMP",
        help="Path to save the trained models.",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=40,
        help="Number of epochs to train.",
    )
    parser.add_argument(
        "--n_workers",
        default=5,
        type=int,
        help="number of data loading workers.",
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="mini-batch size, decrease this if out of memory",
    )
    parser.add_argument(
        "--overwrite_model",
        type=bool_arg,
        default=True,
        help="If True then overwrite the save model if it already exists in the save path.",
    )
    parser.add_argument(
        "--overwrite_depth_exam",
        type=bool_arg,
        default=True,
        help="If True then overwrite the depth examination if it already exists in the save path.",
    )
    parser.add_argument(
        "--update_depth_calib",
        type=bool_arg,
        default=True,
        help="If True then update the depth scale in the model info file based on the depth examination.",
    )
    parser.add_argument(
        "--depth_calib_method",
        type=str,
        choices=["affine", "linear", "none"],
        default="affine",
        help="Method to use for depth scale calibration.",
    )
    parser.add_argument(
        "--debug_mode",
        type=bool_arg,
        default=True,
        help="If True, then use a small dataset and 1 epoch for debugging",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility, if 0 then use the current time as the seed.",
    )

    args = parser.parse_args()
    train_method = args.train_method
    n_ref_imgs = args.n_ref_imgs

    # Set the reference frames time shifts w.r.t.the target frame (-n_ref_imgs, ..., -1)
    ref_frame_shifts = np.arange(-n_ref_imgs, 0).tolist()

    # The size of the depth maps to use.
    depth_map_size = (352, 352)  # we use 352x352 as in the pre-trained FCB-Former model

    print(f"args={args}")
    n_workers = args.n_workers
    torch.cuda.empty_cache()

    random_seed = set_rand_seed(args.seed)
    dataset_path = Path(args.dataset_path)
    save_model_path = Path(args.path_to_save_models)

    load_gt_depth = train_method in {"GT_pose_and_depth", "GT_depth_only"}
    load_gt_pose = train_method in {"GT_pose_and_depth"}

    model_info = ModelInfo(
        depth_model_name=args.depth_model_name,
        egomotion_model_name=args.egomotion_model_name,
        ref_frame_shifts=ref_frame_shifts,
        depth_map_size=depth_map_size,
        model_description=f"The training script args: {args}, random_seed: {random_seed}",
    )
    save_model_info(
        save_dir_path=save_model_path,
        model_info=model_info,
    )

    # The parameters for generating the training samples:
    n_epochs = args.n_epochs
    n_sample_lim = 0  # if 0 then use all the samples in the dataset
    n_scenes_lim = 0  # if 0 then use all the scenes in the dataset for depth examination

    if args.debug_mode:
        print("Running in debug mode!!!!")
        n_sample_lim = 30
        n_scenes_lim = 1
        n_epochs = 1
        save_model_path = save_model_path / "_debug_"
        n_workers = 0  # for debugging
        torch.autograd.set_detect_anomaly(True)

    # dataset split
    dataset_path = Path(args.dataset_path)
    print(f"Loading dataset from {dataset_path}")

    train_scenes_paths, val_scenes_paths = get_scenes_dataset_random_split(
        dataset_path=dataset_path,
        validation_ratio=args.validation_ratio,
    )

    # Crate the datasets (suited for the model_info)
    # training set
    train_set = ScenesDataset(
        scenes_paths=train_scenes_paths,
        model_info=model_info,
        dataset_type="train",
        n_sample_lim=n_sample_lim,
        load_gt_depth=load_gt_depth,
        load_gt_pose=load_gt_pose,
        n_scenes_lim=n_scenes_lim,
    )
    # validation set
    validation_dataset = ScenesDataset(
        scenes_paths=val_scenes_paths,
        model_info=model_info,
        dataset_type="val",
        n_sample_lim=n_sample_lim,
        load_gt_depth=True,
        load_gt_pose=True,
        n_scenes_lim=n_scenes_lim,
    )

    # data loaders
    # training loader
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
        persistent_workers=n_workers > 0,
    )
    # validation loader
    val_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
        persistent_workers=n_workers > 0,
    )

    # Run training:
    train_runner = NetTrainer(
        save_model_path=save_model_path,
        train_loader=train_loader,
        val_loader=val_loader,
        model_info=model_info,
        load_depth_model_path=get_path(args.load_depth_model_path),
        load_egomotion_model_path=get_path(args.load_egomotion_model_path),
        n_epochs=n_epochs,
        run_name="",
    )
    train_runner.train()

    # --------------------------------------------------------------------------------------------------------------------
    # Run depth examination to calibrate the depth scale in the
    model_info = DepthExaminer(
        dataset_path=dataset_path,
        model_path=save_model_path,
        model_info=model_info,
        depth_calib_method=args.depth_calib_method,
        n_scenes_lim=n_scenes_lim,
        save_exam_path=save_model_path / "depth_exam_pre_calib",
        save_overwrite=args.overwrite_depth_exam,
    ).run()

    # --------------------------------------------------------------------------------------------------------------------
    if args.update_depth_calib:
        # update the model info file with the new depth_calib value:
        save_model_info(
            save_dir_path=save_model_path,
            model_info=model_info,
        )


    # --------------------------------------------------------------------------------------------------------------------
    # Run depth examination after updating the depth scale \ calibration: (only for small number of frames)
    DepthExaminer(
        dataset_path=dataset_path,
        model_path=save_model_path,
        model_info=model_info,
        depth_calib_method="none",
        n_scenes_lim=5,
        n_frames_lim=5,
        save_exam_path=save_model_path / "depth_exam_post_calib",
        save_overwrite=args.overwrite_depth_exam,
    ).run()


# ---------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------------------------------------------------
