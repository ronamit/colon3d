import argparse
from pathlib import Path

from colon3d.run_on_sim_dataset import SlamOnDatasetRunner
from colon3d.util.general_util import (
    ArgsHelpFormatter,
    Tee,
    bool_arg,
    delete_incomplete_run_dirs,
    save_run_info,
    save_unified_results_table,
)

# --------------------------------------------------------------------------------------------------------------------
""" Notes:
* Run run_data_prep.py first to generate the dataset of cases with randomly generated targets added to the original scenes.
* You can run several instances of this script in parallel, if setting  delete_incomplete_run_dirs == False, overwrite_results == False.
# *  To run an instance of the script using specific CUDA device (e.g. 0), use the following command:
    CUDA_VISIBLE_DEVICES=0 python -m colon3d.run_col_nav_all  ....
"""
# note: if you want to use the ground truth depth maps, you need to change depth_maps_source to "ground_trutg"
# "EndoSFM_GTD" is still and estimate, but it was trained with GT depth maps
# --------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(formatter_class=ArgsHelpFormatter)

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data_gcp/datasets/ColNav/TestData21",
        help="Path to the dataset of scenes (not raw data, i.e., output of run_data_prep.py ).",
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
        default="data_gcp/models",
    )
    parser.add_argument(
        "--overwrite_results",
        type=bool_arg,
        default=False,
        help="If True then the results folders will be overwritten if they already exists",
    )
    parser.add_argument(
        "--delete_incomplete_run_dirs",
        type=bool_arg,
        default=False,
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
    overwrite_results = args.overwrite_results
    models_base_path = Path(args.models_base_path)
    dataset_path = Path(args.dataset_path)
    # base path to save the algorithm runs results:
    base_results_path = Path(args.results_base_path)
    exp_list = args.exp_list
    # --------------------------------------------------------------------------------------------------------------------

    if args.debug_mode:
        n_cases_lim = 1  # num cases to run the algorithm on
        n_frames_lim = 25  # num frames to run the algorithm on from each scene.
        base_results_path = base_results_path / "debug"
    else:
        n_cases_lim = 0  # no limit
        n_frames_lim = 0  # no limit
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
            "n_scenes_lim": n_cases_lim,
            "save_overwrite": overwrite_results,
        }
        save_unified_results_table(base_results_path)
        # --------------------------------------------------------------------------------------------------------------------
        # the tuned EndoSFM monocular depth and egomotion estimation, with no bundle adjustment
        # --------------------------------------------------------------------------------------------------------------------
        exp_name = "no_BA_with_EndoSFM_tuned"
        if not exp_list or exp_name in exp_list:
            SlamOnDatasetRunner(
                dataset_path=dataset_path,
                save_path=base_results_path / exp_name,
                depth_maps_source="online_estimates",
                egomotions_source="online_estimates",
                depth_and_egomotion_method="EndoSFM",
                depth_and_egomotion_model_path=models_base_path / "EndoSFM_tuned_v3",
                alg_settings_override={"use_bundle_adjustment": False},
                **common_args,
            ).run()
        save_unified_results_table(base_results_path)
        
        # --------------------------------------------------------------------------------------------------------------------
        # the tuned EndoSFM monocular depth and egomotion estimation, with  bundle adjustment
        # --------------------------------------------------------------------------------------------------------------------
        exp_name = "BA_with_EndoSFM_tuned"
        if not exp_list or exp_name in exp_list:
            SlamOnDatasetRunner(
                dataset_path=dataset_path,
                save_path=base_results_path / exp_name,
                depth_maps_source="online_estimates",
                egomotions_source="online_estimates",
                depth_and_egomotion_method="EndoSFM",
                depth_and_egomotion_model_path=models_base_path / "EndoSFM_tuned_v3",
                alg_settings_override={"use_bundle_adjustment": True},
                **common_args,
            ).run()
        save_unified_results_table(base_results_path)
        # --------------------------------------------------------------------------------------------------------------------
        # Bundle-adjustment, with the tuned EndoSFM monocular depth and egomotion estimation
        # --------------------------------------------------------------------------------------------------------------------
        exp_name = "BA_with_EndoSFM_tuned"
        if not exp_list or exp_name in exp_list:
            SlamOnDatasetRunner(
                dataset_path=dataset_path,
                save_path=base_results_path / exp_name,
                depth_maps_source="online_estimates",
                egomotions_source="online_estimates",
                depth_and_egomotion_method="EndoSFM",
                depth_and_egomotion_model_path=models_base_path / "EndoSFM_tuned_v3",
                **common_args,
            ).run()
        save_unified_results_table(base_results_path)
        # --------------------------------------------------------------------------------------------------------------------
        #  Bundle-adjustment, using the ground truth depth maps no egomotions
        # --------------------------------------------------------------------------------------------------------------------
        exp_name = "BA_with_GT_depth_no_ego"
        if not exp_list or exp_name in exp_list:
            SlamOnDatasetRunner(
                dataset_path=dataset_path,
                save_path=base_results_path / exp_name,
                depth_maps_source="ground_truth",
                egomotions_source="none",
                **common_args,
            ).run()

        save_unified_results_table(base_results_path)
        # --------------------------------------------------------------------------------------------------------------------
        # Bundle-adjustment, without monocular depth and egomotion estimation
        # --------------------------------------------------------------------------------------------------------------------
        exp_name = "BA_no_depth_no_ego"
        if not exp_list or exp_name in exp_list:
            SlamOnDatasetRunner(
                dataset_path=dataset_path,
                save_path=base_results_path / exp_name,
                depth_maps_source="none",
                egomotions_source="none",
                **common_args,
            ).run()
        save_unified_results_table(base_results_path)
        
        # --------------------------------------------------------------------------------------------------------------------
        # Bundle-adjustment, with history=2, without monocular depth and egomotion estimation
        # --------------------------------------------------------------------------------------------------------------------
        exp_name = "BA_history_2_no_depth_no_ego"
        if not exp_list or exp_name in exp_list:
            SlamOnDatasetRunner(
                dataset_path=dataset_path,
                save_path=base_results_path / exp_name,
                depth_maps_source="none",
                egomotions_source="none",
                alg_settings_override={"n_last_frames_to_use": 2},
                **common_args,
            ).run()
        save_unified_results_table(base_results_path)
        # --------------------------------------------------------------------------------------------------------------------
        # Bundle-adjustment, with the original EndoSFM monocular depth and egomotion estimation
        # --------------------------------------------------------------------------------------------------------------------
        exp_name = "BA_with_EndoSFM_orig"
        if not exp_list or exp_name in exp_list:
            SlamOnDatasetRunner(
                dataset_path=dataset_path,
                save_path=base_results_path / exp_name,
                depth_maps_source="online_estimates",
                egomotions_source="online_estimates",
                depth_and_egomotion_method="EndoSFM",
                depth_and_egomotion_model_path=models_base_path / "EndoSFM_orig",
                **common_args,
            ).run()
        save_unified_results_table(base_results_path)
        # --------------------------------------------------------------------------------------------------------------------
        # the original EndoSFM monocular depth and egomotion estimation, with no bundle adjustment
        # --------------------------------------------------------------------------------------------------------------------
        exp_name = "no_BA_with_EndoSFM_orig"
        if not exp_list or exp_name in exp_list:
            SlamOnDatasetRunner(
                dataset_path=dataset_path,
                save_path=base_results_path / exp_name,
                depth_maps_source="online_estimates",
                egomotions_source="online_estimates",
                depth_and_egomotion_method="EndoSFM",
                depth_and_egomotion_model_path=models_base_path / "EndoSFM_orig",
                alg_settings_override={"use_bundle_adjustment": False},
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
                depth_and_egomotion_method="none",
                alg_settings_override={"use_bundle_adjustment": False, "use_trivial_nav_aid": True},
                **common_args,
            ).run()
        save_unified_results_table(base_results_path)
        # -------------------------------------------------------------------------------------------------
        # Bundle-adjustment, with the original MonoDepth2 monocular depth and egomotion estimation
        # -------------------------------------------------------------------------------------------------
        exp_name = "BA_with_MonoDepth2_orig"
        if not exp_list or exp_name in exp_list:
            SlamOnDatasetRunner(
                dataset_path=dataset_path,
                save_path=base_results_path / exp_name,
                depth_maps_source="online_estimates",
                egomotions_source="online_estimates",
                depth_and_egomotion_method="MonoDepth2",
                depth_and_egomotion_model_path=models_base_path / "MonoDepth2_orig",
                **common_args,
            ).run()
        save_unified_results_table(base_results_path)
        # --------------------------------------------------------------------------------------------------------------------
        # the original MonoDepth2 monocular depth and egomotion estimation, with no bundle adjustment
        # --------------------------------------------------------------------------------------------------------------------
        exp_name = "no_BA_with_MonoDepth2_orig"
        if not exp_list or exp_name in exp_list:
            SlamOnDatasetRunner(
                dataset_path=dataset_path,
                save_path=base_results_path / exp_name,
                depth_maps_source="online_estimates",
                egomotions_source="online_estimates",
                depth_and_egomotion_method="MonoDepth2",
                depth_and_egomotion_model_path=models_base_path / "monodepth2/MonoDepth2_orig",
                alg_settings_override={"use_bundle_adjustment": False},
                **common_args,
            ).run()
        save_unified_results_table(base_results_path)
        # --------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------------------------------------------------
