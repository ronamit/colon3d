# About

This code loads the simulation output from the Simulator:  The simulator generates single-camera colonoscopic images with ground truth of camera poses.

## References

* [https://github.com/zsustc/colon_reconstruction_dataset](https://github.com/zsustc/colon_reconstruction_dataset)
  that is described in "A Template-based 3D Reconstruction of Colon Structures and Textures from Stereo Colonoscopic Images", Zhang et al., IEEE Transactions on Medical Robotics  2022.
* VR-Caps
* EndoSLAM
* tsdfusion
* pytorch-minimize

## Setup

* Install Conda \ Miniconda
* To create conda environment with the basic required packages use:

```bash
 conda env create -f environment.yml
```

* activate the enviornment

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

* *(optional: for faster surface fuse plot) NVIDIA GPU + [PyCUDA](https://documen.tician.de/pycuda/)*

  * *(Install CUDA, and then:*

    ```bash
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    pip3 install pycuda --user
    ```

## Code use examples

* Here are common examples of how to use the code. See the code for more details on the arguments.
* First, activate the conda environment *(e.g., conda activate py3)* and go to the main project dir (*e.g. ~/repos/colon3d)*.
* Importing a raw dataset of scenes from the unity simulator.

```bash
  python -m colon3d.sim_import.import_from_sim --raw_sim_data_path "data/raw_sim_data/SimData4" --processed_sim_data_path "data/sim_data/SimData4"
```

* Generating examples based on the imported scenes, with randomly simulated tracked targets.

```bash
  python -m colon3d.sim_import.create_scenes_with_tracks --sim_data_path "data/sim_data/SimData11" --path_to_save_examples "data/sim_data/SimData11_with_tracks" --n_examples_per_scene 5
```

* Train depth & egomotion estimators on a held-out set of scenes, starting from pretrained weights.
  Ths training checkpoints will be saved in "saved_models/endo_sfm_opt"

```bash
  python -m endo_sfm.train --name "endo_sfm_opt" --dataset_path "data/sim_data/ScenesForNetsTrain" 
  --pretrained_disp "saved_models/endo_sfm_orig/dispnet_model_best.pt",
  --pretrained_pose "saved_models/endo_sfm_orig/exp_pose_model_best.pt"
```

If out-of-memory error occurs, try to reduce the batch size (e.g. --batch_size 4)

* Run SLAM on a single simulated scene:

```bash
  python -m colon3d.run_slam_on_sim --example_path "data/sim_data/Scene_00009_short/Examples/0000" --save_path "data/sim_data/Scene_00009_short/Examples/0000/Results"
```

* Run SLAM on a dataset of simulated examples:

  ```bash
     python -m colon3d.run_slam_on_sim_dataset --dataset_path  "data/sim_data/SimData8_Examples" --save_path "data/sim_data/SimData8_Examples/Results" --depth_maps_source "none" --egomotions_source "none"
  ```
* Run SLAM on real data example:

```bash
  python -m colon3d.run_slam --example_path "data/my_videos/Example_4" --save_path  "data/my_videos/Example_4/Results" --alg_fov_ratio 0.8 --n_frames_lim 0
```

```bash
  python -m colon3d.run_slam --example_path "data/my_videos/Example_4_rotV2" --save_path  "data/my_videos/Example_4_rotV2/Results" --alg_fov_ratio 0.8 --n_frames_lim 0
```
