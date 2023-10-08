# colon_nav

## About

SLAM using monocular coloscopy videos.

## Setup

* Install Conda \ MiniConda
* (Recommended)
Switch to a faster conflict solver

```bash
conda install -n base conda-libmamba-solver
```

```bash
conda config --set solver libmamba
```

* To create conda environment with the basic required packages, go to the main project dir and run:

```bash
 conda env create -f environment.yml
```

* activate the environment

```bash
 conda activate py_env
```

* Install [PyTorch](https://pytorch.org/get-started/locally/) (tested with 2.1, pip version, CUDA 12.1) e.g. with

```bash
pip3 install -U torch torchvision torchaudio
```

* Install  additional packages:

```bash
 python -m pip install lightning tensorboard tensorboardX pytorch-minimize
```

* Final step:

  * Go to the project main dir (e.g. ~/repos/colon_nav)
  * Run

```bash
pip install -e .
```

## Downloading the data from the cloud

Go to the main project dir and download the Data folder using google cloud CLI.
First, login to your GCP user and project.
To get all the stored data for the project, run:

```bash
gcloud storage cp -r  gs://col_nav/data_gcp  .
```

The folder includes the following subfolders:

* data_gcp/raw_datasets - raw datasets from external sources
* data_gcp/datasets -  datasets, after processing, prepared to be used by the code
* data_gcp/results - saved results of the experiments
* data_gcp/models - saved weights of trained models

## Folder structure

By default, the code will save outputs to the following folders:

* data/datasets - processed datasets
* data/results - results of the experiments
* data/models - weights of trained models

To save an output for long term use, copy it from "data" folder to the "data_gcp" folder, and upload fhe folder to the cloud (after deleting previous results in that path):

```bash
gcloud storage cp -r  data_gcp/PATH_TO_FOLDER  gs://col_nav/data_gcp/PATH_TO_FOLDER
```

## Code usage

* Note that that the all the outputs that was used in the paper are provided in the cloud storage.
* To run a script, activate the conda environment *(e.g., conda activate py_env)* and go to the main project dir (*e.g. ~/repos/colon_nav)*. and run the script using python -m, see examples below.

* We outline below the main run scripts in the code. See the scripts for more details on the run options.

** colon_nav.data_import.import_dataset : Importing raw dataset to the format used by our code.
Examples:

```bash
  python -m colon_nav.data_import.import_dataset  --sim_name "ColonNav" --load_dataset_path "data_gcp/raw_datasets/ColonNav" --save_dataset_path "data/datasets/ColonNav"
```

```bash
  python -m colon_nav.data_import.import_dataset  --sim_name "SimCol3D" --load_dataset_path "data_gcp/raw_datasets/SimCol3D" --save_dataset_path "data/datasets/SimCol3D"
```

* Create unified training dataset: # TODO: add script

* colon_nav.run_ColonNav_exps : Running all experiments on the ColonNav dataset.

* colon_nav.nets.run_train:  Train depth & egomotion estimators using training data.

* colon_nav.run_on_scene: Run the algorithm on a single scene.

* colon_nav.run_on_sim_scene: Run the algorithm on a single simulated scene (that have ground-truth camera pose data).

* colon_nav.run_on_sim_dataset: Run the algorithm on a dataset of simulated examples (that have ground-truth camera pose data).

* Run all experiments on the ColonNav dataset (see file for more details).

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

## Troubleshooting

Exception has occurred: ImportError
libtiff.so.5: cannot open shared object file: No such file or directory
sudo apt-get update
sudo apt-get install libffi-dev
in your Conda environment:
conda install -c conda-forge libffi
conda install -c anaconda libtiff
cd ~/miniconda3/envs/YOUR_ENV/lib
ln -s libtiff.so.6  libtiff.so.5
