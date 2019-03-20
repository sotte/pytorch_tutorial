<img src="notebooks/img/pytorch-logo.png" width="80"> PyTorch Tutorial
================================================================================

This repository contains material to get started with
[PyTorch](https://pytorch.org/) v1.0.

<hr>

Table of Contents
--------------------------------------------------------------------------------

### PART 0 - Foreword
- [Foreword](notebooks/foreword.ipynb) - Why PyTorch and why not? Why this talk?

### PART 1 - Basics
- [PyTorch basics](notebooks/pytorch_basics.ipynb) - tensors, GPU, autograd
  [[colab]](https://colab.research.google.com/github/sotte/pytorch_tutorial/blob/master/notebooks/pytorch_basics.ipynb)
- [Debugging](notebooks/debugging.ipynb)
  [[colab]](https://colab.research.google.com/github/sotte/pytorch_tutorial/blob/master/notebooks/debugging.ipynb)
- [Example: linear regression](notebooks/lin_reg.ipynb)
  [[colab]](https://colab.research.google.com/github/sotte/pytorch_tutorial/blob/master/notebooks/lin_reg.ipynb)
- [Storing and loading models](notebooks/storing_and_loading_models.ipynb)
  [[colab]](https://colab.research.google.com/github/sotte/pytorch_tutorial/blob/master/notebooks/storing_and_loading_models.ipynb)
- [Working with data](notebooks/working_with_data.ipynb) - `Dataset`, `DataLoader`, `Sampler`, `transforms`
  [[colab]](https://colab.research.google.com/github/sotte/pytorch_tutorial/blob/master/notebooks/working_with_data.ipynb)

### PART 2 - Computer Vision
- [Transfer Learning](notebooks/transfer_learning.ipynb)
  [[colab]](https://colab.research.google.com/github/sotte/pytorch_tutorial/blob/master/notebooks/transfer_learning.ipynb)

### PART 3 - Advanced
- [Training Libraries and Visualization](notebooks/training_libraries.ipynb)
  [[colab]](https://colab.research.google.com/github/sotte/pytorch_tutorial/blob/master/notebooks/training_libraries.ipynb)
- [Torch JIT](notebooks/torch_jit.ipynb)
  [[colab]](https://colab.research.google.com/github/sotte/pytorch_tutorial/blob/master/notebooks/torch_jit.ipynb)

### PART -2 - Misc, Cool Applications, Tips, WIP
- [Machine Learning 101 with numpy and PyTorch](notebooks/0x_machine_learning_101.ipynb)
- [PyTorch + GPU in Google's Colab](notebooks/0X_pytorch_in_googles_colab.ipynb)
- [Teacher Forcing](notebooks/0X_teacher_forcing.ipynb)
- [RNNs from Scratch](notebooks/0X_rnn_from_scratch.ipynb)
- [Mean Shift Clustering](notebooks/0X_mean_shift_clustering.ipynb)
- TODO Hooks
- TODO `nn` and `nn.Module`
- TODO Deploy with TF Serving
- TODO init
- TODO PyTorch C++ frontend

### PART -1 - The End
- [The_End](notebooks/the_end.ipynb)

<hr>


Setup
--------------------------------------------------------------------------------

### Requirements

- Python 3.6 or higher
- conda

### Install Dependencies

```bash
# If you have a GPU and CUDA 10
conda env create -f environment_gpu.yml
# If you don't have a GPU
conda env create -f environment_cpu.yml

# activate the conda environment
source activate pytorch_tutorial_123
```

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


Misc
--------------------------------------------------------------------------------

To get the
[jupyter lab table of contents extensions](https://github.com/jupyterlab/jupyterlab-toc)
do the following:
```bash
jupyter labextension install @jupyterlab/toc
```

Prior Versions
--------------------------------------------------------------------------------

- Version of this tutorial for the PyData 2018 conference:
  [[material]](https://github.com/sotte/pytorch_tutorial/tree/pydata2018)
  [[video]](https://nodata.science/pydata-pytorch-tutorial.html)
