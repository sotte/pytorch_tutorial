PyTorch Tutorial
================

This project is going to contain the material of the PyTorch tutorial.

Content
-------
- [00-index](notebooks/00_index.ipynb) Tutorial structure and PyTorch basics
- TODO Transfer learning for Computer Vision
- Intermission / Random Things
  - [0X_mean_shift_clustering](notebooks/0X_mean_shift_clustering.ipynb) Use the GPU to speed up mean shift clustering.
  - TODO RNNs from scratch
  - TODO Teacher forcing

Setup
-----

Please make sure `conda` is installed.
Then:
```bash
# create a conda environment
conda env create -f environment.yml
```
and activate the conda environment.

Download data and models for the tutorial:
```bash
python download_data.py
```
Then you should be ready to go.


To get the [Table of Contenst](https://github.com/ian-r-rose/jupyterlab-toc)
displayed within jupyter lab do the following:
```bash
# install node
conda install -c conda-forge nodejs
# install the toc extension
jupyter labextension install jupyterlab-toc
```
