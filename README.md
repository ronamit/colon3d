# Colon3D

## About

This code loads the simulation output from the Simulator:  The simulator generates single-camera colonoscopic images with ground truth of camera poses.

## Setup

* Install Conda \ MiniConda
* To create conda environment with the basic required packages use:

```bash
 conda env create -f environment.yml
```

* activate the environment

```bash
 conda activate py3
```

* Install [PyTorch](https://pytorch.org/get-started/locally/) (tested with 2.0.1)
* Clone [pytorch-minimize](https://github.com/rfeinman/pytorch-minimize) to some location *(e.g. ~/repos)*

```bash
gh repo clone rfeinman/pytorch-minimize
```

* go to the cloned dir *(e.g. ~/repos/pytorch-minimize)* and run setup with

```bash
pip install -e .
```

* Final step:

  * Go to the project main dir (e.g. ~/repos/colon3d)
  * Run

```bash
pip install -e .
```

## Optional install for faster 3D fuse plot

* (optional: for faster surface fuse plot) NVIDIA GPU + [PyCUDA](https://documen.tician.de/pycuda/)*
* Install CUDA, and then:

  ```bash
  export PATH=/usr/local/cuda/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
  pip3 install pycuda --user
  ```

## Importing the datasets

* Download the dataset from [Zhang22](https://github.com/zsustc/colon_reconstruction_dataset).
  If download fails, try to download each case folder separately.
* Extract all the Case `<number>`  directories (1 to 15)  to a some path, e.g. `<main project dir>`\data\raw_sim_data\Zhang22
* Run the import script:

```bash
python -m colon3d.sim_import.sim_importer --sim_name "Zhang22"  --raw_sim_data_path "data/raw_sim_data/Zhang22" --processed_sim_data_path "data/sim_data/Zhang22"
```

* TODO: add link to our generated \ real dataset.

## Code use examples

* Here are common examples of how to use the code. See the code for more details on the arguments.
* First, activate the conda environment *(e.g., conda activate py3)* and go to the main project dir (*e.g. ~/repos/colon3d)*.
* Importing a raw dataset of scenes from the unity simulator.

```bash
  python -m colon3d.sim_import.import_from_sim --raw_sim_data_path "data/raw_sim_data/SimData4" --processed_sim_data_path "data/sim_data/SimData4"
```

* Generating cases based on the imported scenes, with randomly simulated tracked targets.

```bash
  python -m colon3d.sim_import.create_target_cases --sim_data_path "data/sim_data/SimData11" --path_to_save_cases "data/sim_data/SimData11_cases" --n_cases_per_scene 5
```

* Train depth & egomotion estimators on a held-out set of scenes, starting from pretrained weights.
  Ths training checkpoints will be saved in "saved_models/EndoSFM_tuned"

```bash
  python -m endo_sfm.train --name "EndoSFM_tuned" --dataset_path "data/sim_data/ScenesForNetsTrain"
  --pretrained_disp "saved_models/EndoSFM_orig/DispNet.pt",
  --pretrained_pose "saved_models/EndoSFM_orig/PoseNet.pt"
```

If out-of-memory error occurs, try to reduce the batch size (e.g. --batch_size 4)

* Run the algorithm on a single simulated scene:

```bash
  python -m colon3d.run_on_sim --scene_path "data/sim_data/Scene_00009_short/Examples/0000" --save_path "results/sim_data/Scene_00009_short/Examples/0000/result_new"
```

* Run the algorithm on a dataset of simulated examples:

  ```bash
  python -m colon3d.run_on_sim_dataset --dataset_path  "data/sim_data/SimData8_Examples" --save_path "results/sim_data/SimData8_Examples/result_new" --depth_maps_source "none" --egomotions_source "none"
  ```
  
* Run the algorithm on real data example:

```bash
  python -m colon3d.run_on_scene --scene_path "data/my_videos/Example_4" --save_path  "results/my_videos/Example_4/result_new" --alg_fov_ratio 0.8 --n_frames_lim 0
```

```bash
  python -m colon3d.run_on_scene --scene_path "data/my_videos/Example_4_rotV2" --save_path  "results/my_videos/Example_4_rotV2/result_new" --alg_fov_ratio 0.8 --n_frames_lim 0
```

* Run all experiments with the Zhang22 dataset:

```bash
  python -m colon3d.run_zhang_all --test_dataset_name "Zhang22" --results_name "Zhang22_new" --overwrite_results 1 --overwrite_data 1 --debug_mode 0
```

* Run all experiments on the ColNav dataset:

```bash
  python -m colon3d.run_col_nav_all --test_dataset_name "TestData21" --results_name "ColNav_new" --overwrite_results 1 --overwrite_data 1 --debug_mode 0
```

## References

* [zsustc/colon_reconstruction_dataset](https://github.com/zsustc/colon_reconstruction_dataset)
  that is described in "A Template-based 3D Reconstruction of Colon Structures and Textures from Stereo Colonoscopic Images", Zhang et al., IEEE Transactions on Medical Robotics  2022.
* VR-Caps
* EndoSLAM
* tsdfusion
* pytorch-minimize

## Background sources

* [OpenCV: Camera Calibration and 3D Reconstruction](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html)
* [On transformations.](chemnitz.de/informatik/KI/edu/robotik/ws2017/trans.mat.pdf)
* [OpenCV:L ORB features tutorial](https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html)
* [OpenCV: Feature matching tutorial](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)
* [Scipy: bundle adjustment](https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html)
* [OpenCV: Feature matching + homography](https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html)
