<img src="notebooks/img/pytorch-logo.png" width="80"> PyTorch Tutorial
================================================================================

This repository contains material to get started with
[PyTorch](https://pytorch.org/) v1.7.
It was the base for this
[[pytorch tutorial]](https://nodata.science/pydata-pytorch-tutorial.html)
from PyData Berlin 2018.

<hr>

Table of Contents
--------------------------------------------------------------------------------

### PART 0 - Foreword
- [Foreword](notebooks/foreword.ipynb) - Why PyTorch and why not? Why this talk?

### PART 1 - Basics
- [PyTorch basics](notebooks/pytorch_basics.ipynb) - tensors, GPU, autograd -
  [open in colab](https://colab.research.google.com/github/sotte/pytorch_tutorial/blob/master/notebooks/pytorch_basics.ipynb)
- [Debugging](notebooks/debugging.ipynb) -
  [open in colab](https://colab.research.google.com/github/sotte/pytorch_tutorial/blob/master/notebooks/debugging.ipynb)
- [Example: linear regression](notebooks/lin_reg.ipynb) -
  [open in colab](https://colab.research.google.com/github/sotte/pytorch_tutorial/blob/master/notebooks/lin_reg.ipynb)
- [Storing and loading models](notebooks/storing_and_loading_models.ipynb) -
  [open in colab](https://colab.research.google.com/github/sotte/pytorch_tutorial/blob/master/notebooks/storing_and_loading_models.ipynb)
- [Working with data](notebooks/working_with_data.ipynb) - `Dataset`, `DataLoader`, `Sampler`, `transforms` -
  [open in colab](https://colab.research.google.com/github/sotte/pytorch_tutorial/blob/master/notebooks/working_with_data.ipynb)

### PART 2 - Computer Vision
- [Transfer Learning](notebooks/transfer_learning.ipynb) -
  [open in colab](https://colab.research.google.com/github/sotte/pytorch_tutorial/blob/master/notebooks/transfer_learning.ipynb)

### PART 3 - Misc, Cool Applications, Tips, Advanced
- [Training Libraries and Visualization](notebooks/training_libraries.ipynb)
- [Torch JIT](notebooks/torch_jit.ipynb) -
  [open in colab](https://colab.research.google.com/github/sotte/pytorch_tutorial/blob/master/notebooks/torch_jit.ipynb)
- [Hooks](notebooks/hooks.ipynb) -
  register functions to be called during the forward and backward pass -
  [open in colab](https://colab.research.google.com/github/sotte/pytorch_tutorial/blob/master/notebooks/hooks.ipynb)
- [Machine Learning 101 with numpy and PyTorch](notebooks/machine_learning_101.ipynb) -
  [open in colab](https://colab.research.google.com/github/sotte/pytorch_tutorial/blob/master/notebooks/machine_learning_101.ipynb)
- [PyTorch + GPU in Google's Colab](notebooks/0X_pytorch_in_googles_colab.ipynb)
- [Teacher Forcing](notebooks/0X_teacher_forcing.ipynb)
- [RNNs from Scratch](notebooks/rnn_from_scratch.ipynb) -
  [open in colab](https://colab.research.google.com/github/sotte/pytorch_tutorial/blob/master/notebooks/rnn_from_scratch.ipynb)
- [Mean Shift Clustering](notebooks/mean_shift_clustering.ipynb) -
  [open in colab](https://colab.research.google.com/github/sotte/pytorch_tutorial/blob/master/notebooks/mean_shift_clustering.ipynb)

### PART -2 - WIP and TODO
- TODO `nn` and `nn.Module`
- TODO Deployment
- TODO Deployment with TF Serving
- TODO `nn.init`
- TODO PyTorch C++ frontend

### PART -1 - The End
- [The End](notebooks/the_end.ipynb)

<hr>


Setup
--------------------------------------------------------------------------------

### Requirements

- Python 3.8


### Install Dependencies

```bash
python3.8 -m venv .venv
source .venv/bin/activate.fish
pip install -r requirements.txt
```

#### Optional
Run the following to enable the [jupyter table of contents plugin](https://github.com/jupyterlab/jupyterlab-toc):
```bash
jupyter labextension install @jupyterlab/toc
```
jupyter nbextension enable --py widgetsnbextension

### Download data and models

Download data and models for the tutorial:

```bash
python download_data.py
```

Then you should be ready to go.
Start jupyter lab:

```bash
jupyter lab
```


Prior Versions
--------------------------------------------------------------------------------

- Version of this tutorial for the PyData 2018 conference:
  [[material]](https://github.com/sotte/pytorch_tutorial/tree/pydata2018)
  [[video]](https://nodata.science/pydata-pytorch-tutorial.html)
