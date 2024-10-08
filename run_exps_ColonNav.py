import argparse
import os
from pathlib import Path

from colon_nav.run_on_sim_dataset import SlamOnDatasetRunner
from colon_nav.util.general_util import (
    ArgsHelpFormatter,
    Tee,
    bool_arg,
    delete_incomplete_run_dirs,
    save_run_info,
    save_unified_results_table,
)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# --------------------------------------------------------------------------------------------------------------------
""" Notes:
* Run colon3d.sim_import.create_ColonNav first to generate the dataset of cases with randomly generated targets added to the original scenes.
* You can run several instances of this script in parallel, if setting  delete_incomplete_run_dirs == False, save_overwrite == False.
*  To run an instance of the script using specific CUDA device (e.g. 0), use the following command:
    CUDA_VISIBLE_DEVICES=0 python -m colon3d.run_col_nav_all  ....
"""

# --------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(formatter_class=ArgsHelpFormatter)

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/datasets/ColonNav/Test",
        help="Path to the dataset of scenes (not raw).",
    )
    parser.add_argument(
        "--results_base_path",
        type=str,
        default="data/results/ColonNav",
        help="Base path for the results",
    )
    parser.add_argument(
        "--models_base_path",
        type=str,
        default="data/models",
    )
    parser.add_argument(
        "--save_overwrite",
        type=bool_arg,
        default=False,
        help="If True then the results folders will be overwritten if they already exists",
    )
    parser.add_argument(
        "--delete_incomplete_run_dirs",
        type=bool_arg,
        default=True,
        help="If True then empty results folders will be deleted",
    )
    parser.add_argument(
        "--debug_mode",
        type=bool_arg,
        help="If true, only one scene will be processed, results will be saved to a debug folder",
        default=False,
    )
    parser.add_argument(
        "--exp_list",
        nargs="+",
        type=str,
        default=[],
        help="List of experiments to run, if empty then all experiments will be run",
    )
    args = parser.parse_args()
    save_overwrite = args.save_overwrite
    models_base_path = Path(args.models_base_path)
    dataset_path = Path(args.dataset_path)
    # base path to save the algorithm runs results:
    base_results_path = Path(args.results_base_path)
    exp_list = args.exp_list
    # --------------------------------------------------------------------------------------------------------------------

    if args.debug_mode:
        print("Running in debug mode!!!!")
        n_scenes_lim = 1  # num cases to run the algorithm on
        n_frames_lim = 25  # num frames to run the algorithm on from each scene.
        base_results_path = base_results_path / "debug"
        print_interval = 1  # print progress every X frames
    else:
        n_scenes_lim = 0  # no limit
        n_frames_lim = 0  # no limit
        print_interval = 20  # print progress every X frames
    # ------------------------------------------------------------------------------------------------------------

    with Tee(base_results_path / "log.txt"):  # save the prints to a file
        save_run_info(base_results_path)
        # --------------------------------------------------------------------------------------------------------------------

        if args.delete_incomplete_run_dirs:
            delete_incomplete_run_dirs(base_results_path)
        # --------------------------------------------------------------------------------------------------------------------
        # save summary of existing results:
        save_unified_results_table(base_results_path)
        # --------------------------------------------------------------------------------------------------------------------
        # Run the algorithm on a dataset of simulated examples:
        # --------------------------------------------------------------------------------------------------------------------
        common_args = {
            "save_raw_outputs": False,
            "alg_fov_ratio": 0,
            "n_frames_lim": n_frames_lim,
            "n_scenes_lim": n_scenes_lim,
            "save_overwrite": save_overwrite,
            "load_scenes_with_targets": True,  # Load cases with targets added to the original scenes
            "print_interval": print_interval,
        }

        # --------------------------------------------------------------------------------------------------------------------
        # the (supervised with GT depth) ColonNavModel monocular depth and egomotion estimation, with no bundle adjustment
        # --------------------------------------------------------------------------------------------------------------------
        exp_name = "ColonNavModel_only"
        if not exp_list or exp_name in exp_list:
            SlamOnDatasetRunner(
                dataset_path=dataset_path,
                save_path=base_results_path / exp_name,
                depth_maps_source="online_estimates",
                egomotions_source="online_estimates",
                model_path=models_base_path / "ColonNavModel",
                alg_settings_override={"use_bundle_adjustment": False},
                **common_args,
            ).run()
        save_unified_results_table(base_results_path)

        # --------------------------------------------------------------------------------------------------------------------
        # the (supervised with GT depth) tuned ColonNavModel monocular depth and egomotion estimation, with bundle adjustment
        # --------------------------------------------------------------------------------------------------------------------
        exp_name = "BA_with_ColonNavModel"
        if not exp_list or exp_name in exp_list:
            SlamOnDatasetRunner(
                dataset_path=dataset_path,
                save_path=base_results_path / exp_name,
                depth_maps_source="online_estimates",
                egomotions_source="online_estimates",
                model_path=models_base_path / "ColonNavModel",
                alg_settings_override={"use_bundle_adjustment": True},
                **common_args,
            ).run()
        save_unified_results_table(base_results_path)

        # --------------------------------------------------------------------------------------------------------------------
        # Bundle-adjustment, without monocular depth and egomotion estimation
        # --------------------------------------------------------------------------------------------------------------------
        exp_name = "BA_only"
        if not exp_list or exp_name in exp_list:
            SlamOnDatasetRunner(
                dataset_path=dataset_path,
                save_path=base_results_path / exp_name,
                depth_maps_source="none",
                egomotions_source="none",
                alg_settings_override={"use_bundle_adjustment": True},
                **common_args,
            ).run()
        save_unified_results_table(base_results_path)

        # --------------------------------------------------------------------------------------------------------------------
        # the (supervised with GT depth) EndoSFM monocular depth and egomotion estimation, with no bundle adjustment
        # --------------------------------------------------------------------------------------------------------------------
        exp_name = "EndoSFM_only"
        if not exp_list or exp_name in exp_list:
            SlamOnDatasetRunner(
                dataset_path=dataset_path,
                save_path=base_results_path / exp_name,
                depth_maps_source="online_estimates",
                egomotions_source="online_estimates",
                model_path=models_base_path / "EndoSFM_orig",
                alg_settings_override={"use_bundle_adjustment": False},
                **common_args,
            ).run()
        save_unified_results_table(base_results_path)

        # --------------------------------------------------------------------------------------------------------------------
        # the (supervised with GT depth) tuned EndoSFM monocular depth and egomotion estimation, with bundle adjustment
        # --------------------------------------------------------------------------------------------------------------------
        exp_name = "BA_with_EndoSFM"
        if not exp_list or exp_name in exp_list:
            SlamOnDatasetRunner(
                dataset_path=dataset_path,
                save_path=base_results_path / exp_name,
                depth_maps_source="online_estimates",
                egomotions_source="online_estimates",
                model_path=models_base_path / "EndoSFM_orig",
                alg_settings_override={"use_bundle_adjustment": True},
                **common_args,
            ).run()
        save_unified_results_table(base_results_path)

        # --------------------------------------------------------------------------------------------------------------------
        # Trivial navigation - No bundle-adjustment and no estimations - just use the track's last seen angle
        # --------------------------------------------------------------------------------------------------------------------
        exp_name = "trivial_nav"
        if not exp_list or exp_name in exp_list:
            SlamOnDatasetRunner(
                dataset_path=dataset_path,
                save_path=base_results_path / exp_name,
                depth_maps_source="none",
                egomotions_source="none",
                alg_settings_override={"use_bundle_adjustment": False, "use_trivial_nav_aid": True},
                **common_args,
            ).run()
        save_unified_results_table(base_results_path)
# --------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------------------------------------------------



# # --------------------------------------------------------------------------------------------------------------------
#         # the (supervised with GT depth) tuned EndoSFM monocular depth and egomotion estimation, with no bundle adjustment
#         # --------------------------------------------------------------------------------------------------------------------
#         exp_name = "no_BA_with_EndoSFM_GTD_tuned"
#         if not exp_list or exp_name in exp_list:
#             SlamOnDatasetRunner(
#                 dataset_path=dataset_path,
#                 save_path=base_results_path / exp_name,
#                 depth_maps_source="online_estimates",
#                 egomotions_source="online_estimates",
#                 model_path=models_base_path / "EndoSFM_GTD",
#                 alg_settings_override={"use_bundle_adjustment": False},
#                 **common_args,
#             ).run()
#         save_unified_results_table(base_results_path)

#         # --------------------------------------------------------------------------------------------------------------------
#         # the (supervised with GT depth) tuned EndoSFM monocular depth and egomotion estimation, with bundle adjustment
#         # --------------------------------------------------------------------------------------------------------------------
#         exp_name = "BA_with_EndoSFM_GTD_tuned"
#         if not exp_list or exp_name in exp_list:
#             SlamOnDatasetRunner(
#                 dataset_path=dataset_path,
#                 save_path=base_results_path / exp_name,
#                 depth_maps_source="online_estimates",
#                 egomotions_source="online_estimates",
#                 model_path=models_base_path / "EndoSFM_GTD",
#                 alg_settings_override={"use_bundle_adjustment": True},
#                 **common_args,
#             ).run()
#         save_unified_results_table(base_results_path)

#         # --------------------------------------------------------------------------------------------------------------------
#         # the tuned EndoSFM monocular depth and egomotion estimation, with no bundle adjustment
#         # --------------------------------------------------------------------------------------------------------------------
#         exp_name = "no_BA_with_EndoSFM_tuned"
#         if not exp_list or exp_name in exp_list:
#             SlamOnDatasetRunner(
#                 dataset_path=dataset_path,
#                 save_path=base_results_path / exp_name,
#                 depth_maps_source="online_estimates",
#                 egomotions_source="online_estimates",
#                 model_path=models_base_path / "EndoSFM_tuned",
#                 alg_settings_override={"use_bundle_adjustment": False},
#                 **common_args,
#             ).run()
#         save_unified_results_table(base_results_path)

#         # --------------------------------------------------------------------------------------------------------------------
#         # the tuned EndoSFM monocular depth and egomotion estimation, with  bundle adjustment
#         # --------------------------------------------------------------------------------------------------------------------
#         exp_name = "BA_with_EndoSFM_tuned"
#         if not exp_list or exp_name in exp_list:
#             SlamOnDatasetRunner(
#                 dataset_path=dataset_path,
#                 save_path=base_results_path / exp_name,
#                 depth_maps_source="online_estimates",
#                 egomotions_source="online_estimates",
#                 model_path=models_base_path / "EndoSFM_tuned",
#                 alg_settings_override={"use_bundle_adjustment": True},
#                 **common_args,
#             ).run()
#         save_unified_results_table(base_results_path)
#         # --------------------------------------------------------------------------------------------------------------------
#         # Bundle-adjustment, with the tuned EndoSFM monocular depth and egomotion estimation
#         # --------------------------------------------------------------------------------------------------------------------
#         exp_name = "BA_with_EndoSFM_tuned"
#         if not exp_list or exp_name in exp_list:
#             SlamOnDatasetRunner(
#                 dataset_path=dataset_path,
#                 save_path=base_results_path / exp_name,
#                 depth_maps_source="online_estimates",
#                 egomotions_source="online_estimates",
#                 model_path=models_base_path / "EndoSFM_tuned",
#                 **common_args,
#             ).run()
#         save_unified_results_table(base_results_path)
#         # --------------------------------------------------------------------------------------------------------------------
#         #  Bundle-adjustment, using the ground truth depth maps no egomotions
#         # --------------------------------------------------------------------------------------------------------------------
#         exp_name = "BA_with_GT_depth_no_ego"
#         if not exp_list or exp_name in exp_list:
#             SlamOnDatasetRunner(
#                 dataset_path=dataset_path,
#                 save_path=base_results_path / exp_name,
#                 depth_maps_source="ground_truth",
#                 egomotions_source="none",
#                 **common_args,
#             ).run()

#         save_unified_results_table(base_results_path)
#         # --------------------------------------------------------------------------------------------------------------------
#         # Bundle-adjustment, without monocular depth and egomotion estimation
#         # --------------------------------------------------------------------------------------------------------------------
#         exp_name = "BA_no_depth_no_ego"
#         if not exp_list or exp_name in exp_list:
#             SlamOnDatasetRunner(
#                 dataset_path=dataset_path,
#                 save_path=base_results_path / exp_name,
#                 depth_maps_source="none",
#                 egomotions_source="none",
#                 **common_args,
#             ).run()
#         save_unified_results_table(base_results_path)

#         # --------------------------------------------------------------------------------------------------------------------
#         # Bundle-adjustment, with history=2, without monocular depth and egomotion estimation
#         # --------------------------------------------------------------------------------------------------------------------
#         exp_name = "BA_history_2_no_depth_no_ego"
#         if not exp_list or exp_name in exp_list:
#             SlamOnDatasetRunner(
#                 dataset_path=dataset_path,
#                 save_path=base_results_path / exp_name,
#                 depth_maps_source="none",
#                 egomotions_source="none",
#                 alg_settings_override={"n_last_frames_to_use": 2},
#                 **common_args,
#             ).run()
#         save_unified_results_table(base_results_path)
#         # --------------------------------------------------------------------------------------------------------------------
#         # Bundle-adjustment, with the original EndoSFM monocular depth and egomotion estimation
#         # --------------------------------------------------------------------------------------------------------------------
#         exp_name = "BA_with_EndoSFM_orig"
#         if not exp_list or exp_name in exp_list:
#             SlamOnDatasetRunner(
#                 dataset_path=dataset_path,
#                 save_path=base_results_path / exp_name,
#                 depth_maps_source="online_estimates",
#                 egomotions_source="online_estimates",
#                 model_path=models_base_path / "EndoSFM_orig",
#                 **common_args,
#             ).run()
#         save_unified_results_table(base_results_path)
#         # --------------------------------------------------------------------------------------------------------------------
#         # the original EndoSFM monocular depth and egomotion estimation, with no bundle adjustment
#         # --------------------------------------------------------------------------------------------------------------------
#         exp_name = "no_BA_with_EndoSFM_orig"
#         if not exp_list or exp_name in exp_list:
#             SlamOnDatasetRunner(
#                 dataset_path=dataset_path,
#                 save_path=base_results_path / exp_name,
#                 depth_maps_source="online_estimates",
#                 egomotions_source="online_estimates",
#                 model_path=models_base_path / "EndoSFM_orig",
#                 alg_settings_override={"use_bundle_adjustment": False},
#                 **common_args,
#             ).run()
#         save_unified_results_table(base_results_path)
#         # --------------------------------------------------------------------------------------------------------------------
#         # Trivial navigation - No bundle-adjustment and no estimations - just use the track's last seen angle
#         # --------------------------------------------------------------------------------------------------------------------
#         exp_name = "trivial_nav"
#         if not exp_list or exp_name in exp_list:
#             SlamOnDatasetRunner(
#                 dataset_path=dataset_path,
#                 save_path=base_results_path / exp_name,
#                 depth_maps_source="none",
#                 egomotions_source="none",
#                 alg_settings_override={"use_bundle_adjustment": False, "use_trivial_nav_aid": True},
#                 **common_args,
#             ).run()
#         save_unified_results_table(base_results_path)
#         # -------------------------------------------------------------------------------------------------
#         # Bundle-adjustment, with the original MonoDepth2 monocular depth and egomotion estimation
#         # -------------------------------------------------------------------------------------------------
#         exp_name = "BA_with_MonoDepth2_orig"
#         if not exp_list or exp_name in exp_list:
#             SlamOnDatasetRunner(
#                 dataset_path=dataset_path,
#                 save_path=base_results_path / exp_name,
#                 depth_maps_source="online_estimates",
#                 egomotions_source="online_estimates",
#                 model_path=models_base_path / "MonoDepth2_orig",
#                 **common_args,
#             ).run()
#         save_unified_results_table(base_results_path)
#         # --------------------------------------------------------------------------------------------------------------------
#         # the original MonoDepth2 monocular depth and egomotion estimation, with no bundle adjustment
#         # --------------------------------------------------------------------------------------------------------------------
#         exp_name = "no_BA_with_MonoDepth2_orig"
#         if not exp_list or exp_name in exp_list:
#             SlamOnDatasetRunner(
#                 dataset_path=dataset_path,
#                 save_path=base_results_path / exp_name,
#                 depth_maps_source="online_estimates",
#                 egomotions_source="online_estimates",
#                 model_path=models_base_path / "MonoDepth2_orig",
#                 alg_settings_override={"use_bundle_adjustment": False},
#                 **common_args,
#             ).run()
#         save_unified_results_table(base_results_path)
#         # --------------------------------------------------------------------------------------------------------------------


# # --------------------------------------------------------------------------------------------------------------------


# if __name__ == "__main__":
#     main()

# # --------------------------------------------------------------------------------------------------------------------
