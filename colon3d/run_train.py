import argparse
import os
from pathlib import Path

import torch
import yaml

from colon3d.examine_depths import DepthExaminer
from colon3d.net_train import endosfm_transforms, md2_transforms
from colon3d.net_train.endosfm_trainer import EndoSFMTrainer
from colon3d.net_train.md2_trainer import MonoDepth2Trainer
from colon3d.net_train.scenes_dataset import ScenesDataset, get_scenes_dataset_random_split
from colon3d.util.general_util import ArgsHelpFormatter, bool_arg, save_dict_to_yaml, set_rand_seed
from endo_sfm.utils import save_model_info

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
        "--method",
        type=str,
        default="MonoDepth2_GTPD",
        help="Method to use for depth and egomotion estimation. "
        "Options: MonoDepth2 (self-supervised) | MonoDepth2_GTD (uses ground-truth depth labels) | MonoDepth2_GTPD (uses ground-truth depth + pose labels)"
        "| EndoSFM (self-supervised) | EndoSFM_GTD (uses ground-truth depth labels) | EndoSFM_GTPD (uses ground-truth depth + pose labels).",
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="data_gcp/models/MonoDepth2_orig",  # MonoDepth2_orig | EndoSFM_orig
        help="Path to the pretrained model.",
    )
    parser.add_argument(
        "--path_to_save_model",
        type=str,
        default="data_gcp/models/MonoDepth2_GTPD",
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=20,
        help="Number of epochs to train.",
    )
    parser.add_argument("--n_workers", default=1, type=int, help="number of data loading worker. The current implementation is not thread-safe, so use n_workers=0 to run in the main process.")
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="mini-batch size, decrease this if out of memory",
    )
    parser.add_argument(
        "--subsample_min",
        type=int,
        default=1,
        help="Minimum subsample factor for generating training examples.",
    )
    parser.add_argument(
        "--subsample_max",
        type=int,
        default=5,
        help="Maximum subsample factor for generating training examples.",
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
        default=False,
        help="If True, then use a small dataset and 1 epoch for debugging",
    )
    parser.add_argument(
        "--empty_cache",
        type=bool_arg,
        default=True,
        help="If True, then empty the cache after each epoch (to avoid cuda out of memory error)",
    )
    args = parser.parse_args()
    print(f"args={args}")
    n_workers = args.n_workers
    # Set multiprocessing start method to spawn (to avoid error in DataLoader):
    torch.multiprocessing.set_start_method("spawn")
    if args.empty_cache:
        torch.cuda.empty_cache()

    rand_seed = 0  # random seed for reproducibility
    set_rand_seed(rand_seed)
    dataset_path = Path(args.dataset_path)
    path_to_save_model = Path(args.path_to_save_model)
    pretrained_model_path = Path(args.pretrained_model_path)
    method = args.method

    # load pretrained model info:
    pretrained_model_info_path = pretrained_model_path / "model_info.yaml"
    with pretrained_model_info_path.open("r") as f:
        pretrained_model_info = yaml.load(f, Loader=yaml.FullLoader)
    # the input image size:
    feed_height = pretrained_model_info["feed_height"]
    feed_width = pretrained_model_info["feed_width"]

    if method in {"EndoSFM", "EndoSFM_GTD", "EndoSFM_GTPD"}:
        model_name = "EndoSFM"
    elif method in {"MonoDepth2", "MonoDepth2_GTD", "MonoDepth2_GTPD"}:
        model_name = "MonoDepth2"
    else:
        raise ValueError(f"Unknown method: {method}")

    # methods that use ground-truth depth labels:
    load_gt_depth = method in {"EndoSFM_GTD", "EndoSFM_GTPD", "MonoDepth2_GTD", "MonoDepth2_GTPD"}

    # methods that use ground-truth pose labels:
    load_gt_pose = method in {"EndoSFM_GTPD", "MonoDepth2_GTPD"}

    # The parameters for generating the training samples:
    # for each training example, we randomly a subsample factor to set the frame number between frames in the example (to get wider range of baselines \ ego-motions between the frames)
    subsample_min = args.subsample_min
    subsample_max = args.subsample_max

    path_to_save_depth_exam = path_to_save_model / "depth_exam"

    n_epochs = args.n_epochs
    n_sample_lim = 0  # if 0 then use all the samples in the dataset
    n_scenes_lim = 0  # if 0 then use all the scenes in the dataset for depth examination

    if args.debug_mode:
        print("Running in debug mode!!!!")
        n_sample_lim = 100  # limit the number of samples per epoch (must be more than the batch size)
        n_epochs = 1  # limit the number of epochs
        path_to_save_model = path_to_save_model / "debug"
        n_scenes_lim = 1
        n_workers = 0  # for easier debugging - use only the main process.

    # dataset split
    dataset_path = Path(args.dataset_path)
    print(f"Loading dataset from {dataset_path}")

    train_scenes_paths, val_scenes_paths = get_scenes_dataset_random_split(
        dataset_path=dataset_path,
        validation_ratio=args.validation_ratio,
    )

    # set data transforms
    if model_name == "EndoSFM":
        train_transform, val_transform = endosfm_transforms.get_transforms(
            feed_height=feed_height,
            feed_width=feed_width,
        )
    elif model_name == "MonoDepth2":
        n_scales_md2 = 4
        train_transform, val_transform = md2_transforms.get_transforms(
            n_scales=n_scales_md2,
            feed_height=feed_height,
            feed_width=feed_width,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # training set
    train_set = ScenesDataset(
        scenes_paths=train_scenes_paths,
        feed_height=feed_height,
        feed_width=feed_width,
        transform=train_transform,
        subsample_min=args.subsample_min,
        subsample_max=args.subsample_max,
        n_sample_lim=n_sample_lim,
        load_gt_depth=load_gt_depth,
        load_gt_pose=load_gt_pose,
    )

    # validation set
    validation_dataset = ScenesDataset(
        scenes_paths=val_scenes_paths,
        feed_height=feed_height,
        feed_width=feed_width,
        transform=val_transform,
        subsample_min=1,
        subsample_max=1,
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
    )
    # validation loader
    val_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=n_workers,
        drop_last=True,  # to make sure that the batch size is the same for all batches - drop the last batch if it is smaller than the batch size
    )

    # Run training:
    if model_name == "EndoSFM":
        train_runner = EndoSFMTrainer(
            save_path=path_to_save_model,
            train_loader=train_loader,
            val_loader=val_loader,
            pretrained_disp=pretrained_model_path / "DispNet_best.pt",
            pretrained_pose=pretrained_model_path / "PoseNet_best.pt",
            train_with_gt_depth=load_gt_depth,
            train_with_gt_pose=load_gt_pose,
            n_epochs=n_epochs,
            save_overwrite=args.overwrite_model,
        )
    elif model_name == "MonoDepth2":
        train_runner = MonoDepth2Trainer(
            save_path=path_to_save_model,
            train_loader=train_loader,
            val_loader=val_loader,
            feed_height=feed_height,
            feed_width=feed_width,
            n_scales=4,
            load_weights_folder=pretrained_model_path,
            train_with_gt_depth=load_gt_depth,
            train_with_gt_pose=load_gt_pose,
            save_overwrite=args.overwrite_model,
            n_epochs=n_epochs,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    train_runner.run()

    # --------------------------------------------------------------------------------------------------------------------
    # Save model info:
    # set no depth calibration (depth = net_out) and then run the depth examination to calibrate the depth scale:
    depth_calib = {"depth_calib_type": "none", "depth_calib_a": 1, "depth_calib_b": 0}

    model_description = f"Method: {method}, pretrained model: {pretrained_model_path}, n_epochs: {n_epochs}, subsample_min: {subsample_min}, subsample_max: {subsample_max}, dataset_path: {dataset_path}"
    save_model_info(
        save_dir_path=path_to_save_model,
        feed_height=feed_height,
        feed_width=feed_width,
        num_layers=train_runner.num_layers,
        overwrite=True,
        extra_info={"model_description": model_description, "depth_calib": depth_calib},
    )

    # --------------------------------------------------------------------------------------------------------------------
    # Run depth examination to calibrate the depth scale:
    depth_examiner = DepthExaminer(
        dataset_path=dataset_path,
        model_name=model_name,
        model_path=path_to_save_model,
        save_path=path_to_save_depth_exam,
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
        with info_model_path.open("r") as f:
            model_info = yaml.load(f, Loader=yaml.FullLoader)
            model_info.update(depth_calib)
        save_dict_to_yaml(save_path=info_model_path, dict_to_save=model_info)
        print(
            f"Updated model info file {info_model_path} with the new depth calibration value: {depth_calib}.",
        )

    # --------------------------------------------------------------------------------------------------------------------
    # Run depth examination after updating the depth scale \ calibration: (only for small number of frames)
    depth_examiner = DepthExaminer(
        dataset_path=dataset_path,
        model_name=model_name,
        model_path=path_to_save_model,
        depth_calib_method="none",
        save_path=path_to_save_depth_exam / "after_update",
        n_scenes_lim=5,
        n_frames_lim=5,
        save_overwrite=args.overwrite_depth_exam,
    )
    depth_calib = depth_examiner.run()


# ---------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------------------------------------------------
