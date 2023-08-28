import argparse
import os
from pathlib import Path

import torch
import yaml

import colon3d.net_train.md2_trainer as monodepth2_trainer
from colon3d.examine_depths import DepthExaminer
from colon3d.net_train import endo_sfm_transforms, md2_transforms
from colon3d.net_train.endo_sfm_trainer import TrainRunner as endo_sfm_trainer
from colon3d.net_train.scenes_dataset import ScenesDataset, get_scenes_dataset_random_split
from colon3d.util.data_util import get_all_scenes_paths_in_dir
from colon3d.util.general_util import ArgsHelpFormatter, bool_arg, save_dict_to_yaml, set_rand_seed
from endo_sfm.utils import save_model_info

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"  # prevent cuda out of memory error


# -------------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(formatter_class=ArgsHelpFormatter)
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/mnt/disk1/data/sim_data/TrainData22",
        help="Path to the dataset of scenes used for training (not raw data, i.e., output of import_sim_dataset.py.py ).",
    )
    parser.add_argument(
        "--validation_ratio",
        type=float,
        default=0.1,
        help="ratio of the number of scenes in the validation set from entire training set scenes",
    )
    parser.add_argument(
        "--depth_and_egomotion_method",
        type=str,
        choices=["EndoSFM", "MonoDepth2"],
        default="EndoSFM",
        help="Method to use for depth and egomotion estimation.",
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="/mnt/disk1/saved_models/EndoSFM_orig",
        help="Path to the pretrained model.",
    )
    parser.add_argument(
        "--path_to_save_model",
        type=str,
        default="/mnt/disk1/saved_models/EndoSFM_tuned_v3",
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=200,
        help="Number of epochs to train.",
    )
    parser.add_argument("--n_workers", default=4, type=int, help="number of data loading workers")
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="mini-batch size, decrease this if out of memory",
    )
    parser.add_argument(
        "--overwrite_model",
        type=bool_arg,
        default=False,
        help="If True then overwrite the save model if it already exists in the save patb.",
    )
    parser.add_argument(
        "--overwrite_depth_exam",
        type=bool_arg,
        default=True,
        help="If True then overwrite the depth examination if it already exists in the save path.",
    )
    parser.add_argument(
        "--update_depth_scale",
        type=bool_arg,
        default=True,
        help="If True then update the depth scale in the model info file based on the depth examination.",
    )
    parser.add_argument(
        "--debug_mode",
        type=bool_arg,
        default=True,
        help="If True, then use a small dataset and 1 epoch for debugging",
    )
    args = parser.parse_args()
    print(f"args={args}")

    # --------------------------------------------------------------------------------------------------------------------
    rand_seed = 0  # random seed for reproducibility
    set_rand_seed(rand_seed)
    train_dataset_path = Path(args.train_dataset_path)
    path_to_save_model = Path(args.path_to_save_model)
    pretrained_model_path = Path(args.pretrained_model_path)
    depth_and_egomotion_method = args.depth_and_egomotion_method

    # The parameters for generating the training samples:
    # for each training example, we randomly a subsample factor to set the frame number between frames in the example (to get wider range of baselines \ ego-motions between the frames)
    subsample_param = {"type": "uniform", "min": 1, "max": 20}

    path_to_save_depth_exam = path_to_save_model / "depth_exam"

    n_epochs = args.n_epochs
    epoch_size = 0  # if 0 then use all the samples in the dataset
    n_scenes_lim = 0  # if 0 then use all the scenes in the dataset for depth examination

    if args.debug_mode:
        epoch_size = 5  # limit the number of samples per epoch
        n_epochs = 1  # limit the number of epochs
        path_to_save_model = path_to_save_model / "debug"
        n_scenes_lim = 1

    # --------------------------------------------------------------------------------------------------------------------
    # dataset split
    dataset_path = Path(args.dataset_path)
    print(f"Loading dataset from {dataset_path}")

    train_scenes_paths, val_scenes_paths = get_scenes_dataset_random_split(
        dataset_path=dataset_path,
        validation_ratio=args.validation_ratio,
    )

    # set data transforms
    if depth_and_egomotion_method == "EndoSFM":
        train_transforms = endo_sfm_transforms.get_train_transforms()
        val_transforms = endo_sfm_transforms.get_validation_transforms()
    elif depth_and_egomotion_method == "MonoDepth2":
        train_transforms = md2_transforms.get_train_transforms()
        val_transforms = md2_transforms.get_validation_transforms()
    else:
        raise ValueError(f"Unknown method: {depth_and_egomotion_method}")

    # training set
    train_set = ScenesDataset(
        scenes_paths=train_scenes_paths,
        load_tgt_depth=False,
        transforms=train_transforms,
        subsample_param=subsample_param,
    )

    # validation set
    validation_dataset = ScenesDataset(
        scenes_paths=val_scenes_paths,
        load_tgt_depth=True,
        transforms=val_transforms,
        subsample_param=subsample_param,
    )

    # data loaders
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        pin_memory=True,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        pin_memory=True,
    )

    # Run training:

    if depth_and_egomotion_method == "EndoSFM":
        train_runner = endo_sfm_trainer(
            save_path=path_to_save_model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            pretrained_disp=pretrained_model_path / "DispNet_best.pt",
            pretrained_pose=pretrained_model_path / "PoseNet_best.pt",
            save_overwrite=args.overwrite_model,
            n_epochs=n_epochs,
            epoch_size=epoch_size,
        )
    elif depth_and_egomotion_method == "MonoDepth2":
        train_runner = monodepth2_trainer(
            dataset_path=train_dataset_path,
            save_path=path_to_save_model,
            load_weights_folder=pretrained_model_path,
            subsample_param=subsample_param,
            save_overwrite=args.overwrite_model,
            n_epochs=n_epochs,
            epoch_size=epoch_size,
        )
    else:
        raise ValueError(f"Unknown method: {depth_and_egomotion_method}")
    train_runner.run()

    # --------------------------------------------------------------------------------------------------------------------
    # Save model info:
    # get scene metata for some example scene in the train dataset (should be the same for all scenes):
    scenes_paths = get_all_scenes_paths_in_dir(train_dataset_path, with_targets=False)
    scene_path = scenes_paths[0]
    with (scene_path / "meta_data.yaml").open("r") as f:
        scene_metadata = yaml.load(f, Loader=yaml.FullLoader)

    # set this value initially to 1.0, and then run the depth examination to calibrate the depth scale:
    net_out_to_mm = 1.0

    model_description = f"Method: {depth_and_egomotion_method}, pretrained model: {pretrained_model_path}, n_epochs: {n_epochs}, subsample_param: {subsample_param}, dataset_path: {train_dataset_path}"
    save_model_info(
        save_dir_path=path_to_save_model,
        scene_metadata=scene_metadata,
        disp_resnet_layers=train_runner.disp_resnet_layers,
        pose_resnet_layers=train_runner.pose_resnet_layers,
        overwrite=True,
        extra_info={"model_description": model_description, "net_out_to_mm": net_out_to_mm},
    )

    # --------------------------------------------------------------------------------------------------------------------
    # Run depth examination to calibrate the depth scale:
    depth_examiner = DepthExaminer(
        dataset_path=train_dataset_path,
        depth_and_egomotion_method=depth_and_egomotion_method,
        depth_and_egomotion_model_path=path_to_save_model,
        save_path=path_to_save_depth_exam,
        n_scenes_lim=n_scenes_lim,
        save_overwrite=args.overwrite_depth_exam,
    )
    depth_exam = depth_examiner.run()

    if args.update_depth_scale:
        # the output of the depth network needs to be multiplied by this number to get the depth in mm (based on the analysis of the true depth data in examine_depths.py)
        net_out_to_mm = depth_exam["avg_gt_to_est_depth_ratio"]
        info_model_path = path_to_save_model / "model_info.yaml"
        # update the model info file with the new net_out_to_mm value:
        with info_model_path.open("r") as f:
            model_info = yaml.load(f, Loader=yaml.FullLoader)
            model_info["net_out_to_mm"] = net_out_to_mm
        save_dict_to_yaml(save_path=info_model_path, dict_to_save=model_info)
        print(
            f"Updated model info file {info_model_path} with the calibrated net_out_to_mm value: {net_out_to_mm} [mm]"
        )


# --------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------------------------------------------------
