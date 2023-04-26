# Code

This code loads the simulation output from the Simulator:  [https://github.com/zsustc/colon_reconstruction_dataset](https://github.com/zsustc/colon_reconstruction_dataset)
that is described in "A Template-based 3D Reconstruction of Colon Structures and Textures from Stereo Colonoscopic Images", Zhang et al., IEEE Transactions on Medical Robotics  2022.
The simulator generates colonoscopic images with ground truth of camera poses.
The simulator has option for stereo cameras, but here we assume it was used with one camera.

[Example saved sequence](https://drive.google.com/drive/folders/1ADir7CwF9NTUVIH-1Og2BpBeAf10afYV?usp=sharing)

## Setup

### Prerequisites

* Conda \ Miniconda
* You may need to install this
  ```bash
  conda update --name base conda
  conda install conda-build
  ```
* To create conda envion,ent with the required packages use:

```bash
 conda env create -f environment.yml python=3.9
```

* Install [pytorch-minimize](https://github.com/rfeinman/pytorch-minimize) from source.
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
