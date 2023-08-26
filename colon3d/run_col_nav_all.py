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
# --------------------------------------------------------------------------------------------------------------------

def  run_exp(exp_list: list, exp_name: str, dataset_path: str, base_results_path: str, models_base_path: str, depth_maps_source: str, egomotions_source: str, depth_and_egomotion_method: str, model_name: str, alg_settings_override: dict, common_args: dict):
    if not exp_list or exp_name in exp_list:
        SlamOnDatasetRunner(
            dataset_path=dataset_path,
            save_path=base_results_path / exp_name,
            depth_maps_source=depth_maps_source,
            egomotions_source=egomotions_source,
            depth_and_egomotion_method=depth_and_egomotion_method,
            depth_and_egomotion_model_path=models_base_path / model_name,
            alg_settings_override=alg_settings_override,
            **common_args,
        ).run()
    save_unified_results_table(base_results_path)

# --------------------------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(formatter_class=ArgsHelpFormatter)

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/mnt/disk1/data/sim_data/TestData21",
        help="Path to the dataset of scenes (not raw data, i.e., output of run_data_prep.py ).",
    )
    parser.add_argument(
        "--results_base_path",
        type=str,
        default="/mnt/disk1/results/ColonNav",
        help="Base path for the results",
    )
    parser.add_argument(
        "--models_base_path",
        type=str,
        default="/mnt/disk1/saved_models",
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
        n_cases_lim = 0 # no limit
        n_frames_lim = 0 # no limit
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
        # --------------------------------------------------------------------------------------------------------------------
        # the tuned EndoSFM monocular depth and egomotion estimation, with no bundle adjustment
        # --------------------------------------------------------------------------------------------------------------------
        exp_name = "no_BA_with_EndoSFM_tuned"
        run_exp(
            exp_list=exp_list,
            exp_name=exp_name,
            dataset_path=dataset_path,
            base_results_path=base_results_path,
            models_base_path=models_base_path,
            depth_maps_source="online_estimates",
            egomotions_source="online_estimates",
            depth_and_egomotion_method="EndoSFM",
            model_name="EndoSFM_tuned_v2",
            alg_settings_override={"use_bundle_adjustment": False},
            common_args=common_args,
        )
        # --------------------------------------------------------------------------------------------------------------------
        # Bundle-adjustment, with the tuned EndoSFM monocular depth and egomotion estimation
        # --------------------------------------------------------------------------------------------------------------------
        exp_name = "BA_with_EndoSFM_tuned"
        run_exp(
            exp_list=exp_list,
            exp_name=exp_name,
            dataset_path=dataset_path,
            base_results_path=base_results_path,
            models_base_path=models_base_path,
            depth_maps_source="online_estimates",
            egomotions_source="online_estimates",
            depth_and_egomotion_method="EndoSFM",
            model_name="EndoSFM_tuned_v2",
            common_args=common_args,
        )
        # --------------------------------------------------------------------------------------------------------------------
        #  Bundle-adjustment, using the ground truth depth maps no egomotions
        # --------------------------------------------------------------------------------------------------------------------
        exp_name = "BA_with_GT_depth_no_ego"
        run_exp(
            exp_list=exp_list,
            exp_name=exp_name,
            dataset_path=dataset_path,
            base_results_path=base_results_path,
            models_base_path=models_base_path,
            depth_maps_source="ground_truth",
            egomotions_source="none",
            depth_and_egomotion_method="none",
            common_args=common_args,
        )
        # --------------------------------------------------------------------------------------------------------------------
        # Bundle-adjustment, without monocular depth and egomotion estimation
        # --------------------------------------------------------------------------------------------------------------------
        exp_name = "BA_no_depth_no_ego"
        run_exp(
            exp_list=exp_list,
            exp_name=exp_name,
            dataset_path=dataset_path,
            base_results_path=base_results_path,
            models_base_path=models_base_path,
            depth_maps_source="none",
            egomotions_source="none",
            depth_and_egomotion_method="none",
            common_args=common_args,
        )
        # --------------------------------------------------------------------------------------------------------------------
        # Bundle-adjustment, with the original EndoSFM monocular depth and egomotion estimation
        # --------------------------------------------------------------------------------------------------------------------
        exp_name = "BA_with_EndoSFM_orig"
        run_exp(
            exp_list=exp_list,
            exp_name=exp_name,
            dataset_path=dataset_path,
            base_results_path=base_results_path,
            models_base_path=models_base_path,
            depth_maps_source="online_estimates",
            egomotions_source="online_estimates",
            depth_and_egomotion_method="EndoSFM",
            model_name="EndoSFM_orig",
            common_args=common_args,
        )
        # --------------------------------------------------------------------------------------------------------------------
        # the original EndoSFM monocular depth and egomotion estimation, with no bundle adjustment
        # --------------------------------------------------------------------------------------------------------------------
        exp_name = "no_BA_with_EndoSFM_orig"
        run_exp(
            exp_list=exp_list,
            exp_name=exp_name,
            dataset_path=dataset_path,
            base_results_path=base_results_path,
            models_base_path=models_base_path,
            depth_maps_source="online_estimates",
            egomotions_source="online_estimates",
            depth_and_egomotion_method="EndoSFM",
            model_name="EndoSFM_orig",
            alg_settings_override={"use_bundle_adjustment": False},
            common_args=common_args,
        )
        # --------------------------------------------------------------------------------------------------------------------
        # Trivial navigation - No bundle-adjustment and no estimations - just use the track's last seen angle
        # --------------------------------------------------------------------------------------------------------------------
        exp_name = "trivial_nav"
        run_exp(
            exp_list=exp_list,
            exp_name=exp_name,
            dataset_path=dataset_path,
            base_results_path=base_results_path,
            models_base_path=models_base_path,
            depth_maps_source="none",
            egomotions_source="none",
            depth_and_egomotion_method="none",
            alg_settings_override={"use_bundle_adjustment": False, "use_trivial_nav_aid": True},
            common_args=common_args,
        )
        # -------------------------------------------------------------------------------------------------
        # Bundle-adjustment, with the original MonoDepth2 monocular depth and egomotion estimation
        # -------------------------------------------------------------------------------------------------
        exp_name = "BA_with_MonoDepth2_orig"
        run_exp(
            exp_list=exp_list,
            exp_name=exp_name,
            dataset_path=dataset_path,
            base_results_path=base_results_path,
            models_base_path=models_base_path,
            depth_maps_source="online_estimates",
            egomotions_source="online_estimates",
            depth_and_egomotion_method="MonoDepth2",
            model_name="monodepth2/mono_stereo_640x192_orig",
            common_args=common_args,
        )
        # --------------------------------------------------------------------------------------------------------------------
        # the original MonoDepth2 monocular depth and egomotion estimation, with no bundle adjustment
        # --------------------------------------------------------------------------------------------------------------------
        exp_name = "no_BA_with_MonoDepth2_orig"
        run_exp(
            exp_list=exp_list,
            exp_name=exp_name,
            dataset_path=dataset_path,
            base_results_path=base_results_path,
            models_base_path=models_base_path,
            depth_maps_source="online_estimates",
            egomotions_source="online_estimates",
            depth_and_egomotion_method="MonoDepth2",
            model_name="monodepth2/mono_stereo_640x192_orig",
            alg_settings_override={"use_bundle_adjustment": False},
            common_args=common_args,
        )
        

# --------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------------------------------------------------
