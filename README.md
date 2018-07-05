PyTorch Tutorial
================

This project is going to contain the material of the PyTorch tutorial.

Content
-------
- [00-index](notebooks/00_index.ipynb)


Setup
-----

Please make sure `conda` is installed.

### Manual install
Due to some problems with the environment.yml here are instructions
for a manual installation:
```bash
# create conda environment named ppt
conda new --name ppt
source activate ppt

# Install dependencies
conda install -y matplotlib numpy scipy tensorflow
conda install -y pytorch-cpu torchvision-cpu ignite -c pytorch
conda install -y jupyterlab -c conda-forge
pip install tensorboardX scikit-learn
pip install -e .
```


### Install with environment.yml

Then:
```bash
# create a conda environment
conda env create -f environment.yml
```
activate the conda environment
```bash
source activate pydata_pytorch_tutorial
```
and install the `ppt` package (this project basically)
```bash
pip install -e .
```


### Mac
If you have problems with the dependencies under mac check out this issue:
https://github.com/sotte/pytorch_tutorial/issues/2

```bash
# You might have to use the `environment_mac.yml`
conda env create -f environment_mac.yml
# and manually update freetype and matplotlib
conda update freetype matplotlib
```


### Download data and models
Download data and models for the tutorial:
```bash
python download_data.py
```
Then you should be ready to go.
Start jupyter lab
```bash
jupyter lab
```


### Misc
To get the [Table of Contents](https://github.com/ian-r-rose/jupyterlab-toc)
displayed within jupyter lab do the following:
```bash
# install node
conda install -c conda-forge nodejs
# install the toc extension
jupyter labextension install jupyterlab-toc
```
