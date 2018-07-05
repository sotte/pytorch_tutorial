PyTorch Tutorial
================

This project is going to contain the material of the PyTorch tutorial.

Content
-------
- [00-index](notebooks/00_index.ipynb)


Setup
-----

Please make sure `conda` is installed.
Then:
```bash
# create a conda environment
conda env create -f environment.yml
```
and activate the conda environment.
```bash
source activate pydata_pytorch_tutorial
```
and install the `ppt` package (this project basically)
```bash
pip install -e .
```

Download data and models for the tutorial:
```bash
python download_data.py
```
Then you should be ready to go.
Start jupyter lab
```bash
jupyter lab
```


To get the [Table of Contents](https://github.com/ian-r-rose/jupyterlab-toc)
displayed within jupyter lab do the following:
```bash
# install node
conda install -c conda-forge nodejs
# install the toc extension
jupyter labextension install jupyterlab-toc
```
