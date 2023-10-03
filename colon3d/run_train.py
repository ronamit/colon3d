import argparse
import os
from pathlib import Path

import torch
import yaml

from colon3d.examine_depths import DepthExaminer
from colon3d.net_train.endosfm_trainer import EndoSFMTrainer
from colon3d.net_train.md2_trainer import MonoDepth2Trainer
from colon3d.net_train.scenes_dataset import ScenesDataset, get_scenes_dataset_random_split
from colon3d.net_train.train_utils import ModelInfo, get_default_model_info, save_model_info
from colon3d.util.general_util import ArgsHelpFormatter, bool_arg, set_rand_seed

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"  # prevent cuda out of memory error

"""
If out-of-memory error occurs, try to reduce the batch size (e.g. --batch_size 4)
"""

# turn on anamoly detection for debugging:
torch.autograd.set_detect_anomaly(True)


# -------------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(formatter_class=ArgsHelpFormatter)
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data_gcp/datasets/UnifiedTrain",
        help="Path to the dataset of scenes used for training (not raw data, i.e., output of import_dataset.py.py ).",
    )
    parser.add_argument(
        "--validation_ratio",
        type=float,
        default=0.1,
        help="ratio of the number of scenes in the validation set from entire training set scenes",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="MonoDepth2",
        choices=["MonoDepth2", "EndoSFM"],
        help="Name of the model to train.",
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
        "--pretrained_model_path",
        type=str,
        default="",
        help="Path to the pretrained model. If empty string then use the default ImageNet pretrained weights."
        "Note that a loaded pretrained model must be compatible with the model_name and n_ref_imgs.",
    )
    parser.add_argument(
        "--path_to_save_model",
        type=str,
        default="data_gcp/models/TEMP",
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=20,
        help="Number of epochs to train.",
    )
    parser.add_argument(
        "--n_workers",
        default=9,
        type=int,
        help="number of data loading workers.",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
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

    args = parser.parse_args()
    train_method = args.train_method
    n_ref_imgs = args.n_ref_imgs
    model_name = args.model_name

    print(f"args={args}")
    n_workers = args.n_workers
    torch.cuda.empty_cache()

    rand_seed = 0  # random seed for reproducibility
    set_rand_seed(rand_seed)
    dataset_path = Path(args.dataset_path)
    path_to_save_model = Path(args.path_to_save_model)

    load_gt_depth = train_method in {"GT_pose_and_depth", "GT_depth_only"}
    load_gt_pose = train_method in {"GT_pose_and_depth"}

    if args.pretrained_model_path != "":
        pretrained_model_path = Path(args.pretrained_model_path)
        # load pretrained model info:
        model_info_path = pretrained_model_path / "model_info.yaml"
        with model_info_path.open("r") as f:
            model_info = yaml.load(f, Loader=yaml.FullLoader)
        assert n_ref_imgs == model_info["n_ref_imgs"], "n_ref_imgs must be the same as the pretrained model"
    else:
        # Train from scratch (pretrained on ImageNet)
        model_info = get_default_model_info(model_name)
        pretrained_model_path = None

    # the input image size:
    feed_height = model_info["feed_height"]
    feed_width = model_info["feed_width"]

    # The parameters for generating the training samples:
    n_epochs = args.n_epochs
    n_sample_lim = 0  # if 0 then use all the samples in the dataset
    n_scenes_lim = 0  # if 0 then use all the scenes in the dataset for depth examination

    if args.debug_mode:
        print("Running in debug mode!!!!")
        n_sample_lim = 5
        n_epochs = 1
        path_to_save_model = path_to_save_model / "debug"
        n_scenes_lim = 1
        n_workers = 0  # for debugging

    # dataset split
    dataset_path = Path(args.dataset_path)
    print(f"Loading dataset from {dataset_path}")

    train_scenes_paths, val_scenes_paths = get_scenes_dataset_random_split(
        dataset_path=dataset_path,
        validation_ratio=args.validation_ratio,
    )

    # training set
    train_set = ScenesDataset(
        scenes_paths=train_scenes_paths,
        feed_height=feed_height,
        feed_width=feed_width,
        model_name=model_name,
        dataset_type="train",
        n_ref_imgs=n_ref_imgs,
        n_sample_lim=n_sample_lim,
        load_gt_depth=load_gt_depth,
        load_gt_pose=load_gt_pose,
    )

    # validation set
    validation_dataset = ScenesDataset(
        scenes_paths=val_scenes_paths,
        feed_height=feed_height,
        feed_width=feed_width,
        model_name=model_name,
        dataset_type="val",
        n_ref_imgs=n_ref_imgs,
        n_sample_lim=n_sample_lim,
        load_gt_depth=True,
        load_gt_pose=True,
    )

    # data loaders
    # training loader
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=n_workers,
        drop_last=True,  # to make sure that the batch size is the same for all batches - drop the last batch if it is smaller than the batch size
        pin_memory=True,
        persistent_workers=n_workers > 0,
    )
    # validation loader
    val_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=n_workers,
        drop_last=True,  # to make sure that the batch size is the same for all batches - drop the last batch if it is smaller than the batch size
        pin_memory=True,
        persistent_workers=n_workers > 0,
    )

    # Run training:
    if model_name == "EndoSFM":
        train_runner = EndoSFMTrainer(
            save_path=path_to_save_model,
            train_loader=train_loader,
            val_loader=val_loader,
            pretrained_model_path=pretrained_model_path,
            n_epochs=n_epochs,
            save_overwrite=args.overwrite_model,
        )
    elif model_name == "MonoDepth2":
        train_runner = MonoDepth2Trainer(
            save_path=path_to_save_model,
            train_loader=train_loader,
            val_loader=val_loader,
            n_scales=4,
            pretrained_model_path=pretrained_model_path,
            save_overwrite=args.overwrite_model,
            n_epochs=n_epochs,
        )
    else:
        raise ValueError(f"Unknown method: {model_name}")
    train_runner.run()

    # --------------------------------------------------------------------------------------------------------------------
    # Save model info:
    # set no depth calibration (depth = net_out) and then run the depth examination to calibrate the depth scale:
    model_description = f"The model was trained on:  pretrained_model_path: {pretrained_model_path}, n_epochs: {n_epochs}, dataset_path: {dataset_path}, train_method: {train_method}"
    model_info = ModelInfo(
        model_name=model_name,
        n_ref_imgs=n_ref_imgs,
        feed_height=feed_height,
        feed_width=feed_width,
        num_layers=train_runner.num_layers,
        model_description=model_description,
    )
    save_model_info(
        save_dir_path=path_to_save_model,
        model_info=model_info,
    )

    # --------------------------------------------------------------------------------------------------------------------
    # Run depth examination to calibrate the depth scale:
    depth_examiner = DepthExaminer(
        dataset_path=dataset_path,
        model_path=path_to_save_model,
        save_path=path_to_save_model / "depth_exam_pre_calib",
        depth_calib_method=args.depth_calib_method,
        n_scenes_lim=n_scenes_lim,
        save_overwrite=args.overwrite_depth_exam,
    )
    depth_calib = depth_examiner.run()

    # --------------------------------------------------------------------------------------------------------------------
    if args.update_depth_calib:
        # the output of the depth network needs to transformed with this to get the depth in mm (based on the analysis of the true depth data in examine_depths.py)
        info_model_path = path_to_save_model / "model_info.yaml"
        # update the model info file with the new depth_calib value:
        model_info.depth_calib_type = depth_calib["depth_calib_type"]
        model_info.depth_calib_a = depth_calib["depth_calib_a"]
        model_info.depth_calib_b = depth_calib["depth_calib_b"]
        save_model_info(
            save_dir_path=path_to_save_model,
            model_info=model_info,
        )
        print(
            f"Updated model info file {info_model_path} with the new depth calibration value: {depth_calib}.",
        )

    # --------------------------------------------------------------------------------------------------------------------
    # Run depth examination after updating the depth scale \ calibration: (only for small number of frames)
    depth_examiner = DepthExaminer(
        dataset_path=dataset_path,
        model_path=path_to_save_model,
        depth_calib_method="none",
        save_path=path_to_save_model / "depth_exam_post_calib",
        n_scenes_lim=5,
        n_frames_lim=5,
        save_overwrite=args.overwrite_depth_exam,
    )
    depth_calib = depth_examiner.run()


# ---------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------------------------------------------------
