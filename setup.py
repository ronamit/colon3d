# (this allows simple imports inside the project - see https://stackoverflow.com/a/50194143)
# run setup with `pip install -e .` (from the project root directory)

from setuptools import find_packages, setup

setup(name=["colon3d", "tsdf_fusion", "endo_sfm", "monodepth2", "sc_depth"], version="1.0", packages=find_packages())
