# Code

This code loads the simulation output from the Simulator:  [https://github.com/zsustc/colon_reconstruction_dataset](https://github.com/zsustc/colon_reconstruction_dataset)
that is described in "A Template-based 3D Reconstruction of Colon Structures and Textures from Stereo Colonoscopic Images", Zhang et al., IEEE Transactions on Medical Robotics  2022.
The simulator generates single-camera colonoscopic images with ground truth of camera poses.

## Setup

* Install Conda \ Miniconda
* You may need to install this

  ```bash
  conda update --name base conda
  conda install conda-build
  ```

* To create conda environment with the required packages use:

 conda env create -f environment.yml

<!-- * Install [pytorch-minimize](https://github.com/rfeinman/pytorch-minimize) from source. -->
* *(optional: for faster surface fuse plot) NVIDIA GPU + [PyCUDA](https://documen.tician.de/pycuda/)*

  * *(Install CUDA, and then:*

    ```bash
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    pip3 install pycuda --user
    ```

* Setup:

  * Go to the project root
  * Run

```bash
  conda develop .
```

## Code use examples

* Here are common examples of how to use the code. See the code for more details on the arguments.
* First, activate the conda environment and go to the main project dir.
* Importing a frame sequence output from the unity simulator, with 200 frames limit

```bash
  python -m colon3d.unity_import.import_from_sim --raw_sim_data_path "data/raw_sim_data/Seq_00009" --path_to_save_sequence "data/sim_data/Seq_00009_short" --limit_n_frames 200
```

* Generating examples based on the imported sequence, with true depth maps and ego-motions as the estimates for the SLAM algorithm use.

```bash
  python -m colon3d.unity_import.create_examples --sequence_path "data/sim_data/Seq_00009_short" --n_examples 1 --depth_noise_std_mm 0 --cam_motion_loc_std_mm 0 --cam_motion_rot_std_deg 0
```

* Run SLAM on simulated example:

```bash
  python -m colon3d.run_slam_on_sim --example_path "data/sim_data/Seq_00009_short/Examples/0000" --save_path "data/sim_data/Seq_00009_short/Examples/0000/results" --alg_fov_ratio 0.95
```

* Run SLAM on real data example:

```bash
  python -m colon3d.run_slam --example_path "data/my_videos/Example_4" --save_path  "data/my_videos/Example_4/results" --alg_fov_ratio 0.8 --n_frames_lim 0
```

```bash
  python -m colon3d.run_slam --example_path "data/my_videos/Example_4_rotV2" --save_path  "data/my_videos/Example_4_rotV2/results" --alg_fov_ratio 0.8 --n_frames_lim 0
```
