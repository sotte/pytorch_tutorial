# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ![](img/the_real_reason.png)

# %% [markdown]
# # Foreword
#
# Material for this tutorial is here: https://github.com/sotte/pytorch_tutorial
#
# **Prerequisites:**
# - you have implemented machine learning models yourself
# - you know what deep learning is
# - you have used numpy
# - maybe you have used tensorflow or similar libs
#
# - if you use PyTorch on a daily basis, this tutorial is probably not for you
#
# **Goals:**
# - understand PyTorch concepts
# - be able to use transfer learning in PyTorch
# - be aware of some handy tools/libs

# %% [markdown]
# Note:
# You don't need a GPU to work on this tutorial, but everything is much faster if you have one.
# However, you can use Google's Colab with a GPU and work on this tutorial:
# [PyTorch + GPU in Google's Colab](0X_pytorch_in_googles_colab.ipynb)

# %% [markdown]
# # Agenda
#
# See README.md

# %% [markdown]
# # PyTorch Overview
#
#
# > "PyTorch - Tensors and Dynamic neural networks in Python
# with strong GPU acceleration.
# PyTorch is a deep learning framework for fast, flexible experimentation."
# >
# > -- https://pytorch.org/*
#
# This was the tagline prior to PyTorch 1.0.
# Now it's:
#
# > "PyTorch - From Research To Production
# > 
# > An open source deep learning platform that provides a seamless path from research prototyping to production deployment."

# %% [markdown]
# ## "Build by run" - what is that and why do I care?

# %% [markdown]
# ![](img/dynamic_graph.gif)

# %% [markdown]
# This is a much better explanation of PyTorch (I think)

# %%
import torch
from IPython.core.debugger import set_trace

def f(x):
    res = x + x
    # set_trace()  # <-- OMG! =D
    return res

x = torch.randn(1, 10)
f(x)

# %% [markdown]
# I like pytorch because
# - "it's just stupid python"
# - easy to debug
# - nice and extensible interface
# - research-y feel
# - research is often published as pytorch project

# %% [markdown]
# ## A word about TF
# TF 2 is about to be released.
# - eager by default
# - API cleanup
# - No more `session.run()`, `tf.control_dependencies()`, `tf.while_loop()`, `tf.cond()`, `tf.global_variables_initializer()`, etc.
#
# ## TF and PyTorch
# - static vs dynamic
# - production vs prototyping 

# %% [markdown]
# ## *"The tyranny of choice"*
# - TensorFlow
# - MXNet
# - Keras
# - CNTK
# - Chainer
# - caffe
# - caffe2
# - many many more
#
# All of them a good!
#

# %% [markdown]
# # References
# - Twitter: https://twitter.com/PyTorch
# - Forum: https://discuss.pytorch.org/
# - Tutorials: https://pytorch.org/tutorials/
# - Examples: https://github.com/pytorch/examples
# - API Reference: https://pytorch.org/docs/stable/index.html
# - Torchvision: https://pytorch.org/docs/stable/torchvision/index.html
# - PyTorch Text: https://github.com/pytorch/text
# - PyTorch Audio: https://github.com/pytorch/audio
# - AllenNLP: https://allennlp.org/
# - Object detection/segmentation: https://github.com/facebookresearch/maskrcnn-benchmark
# - Facebook AI Research Sequence-to-Sequence Toolkit written in PyTorch: https://github.com/pytorch/fairseq
# - FastAI http://www.fast.ai/
# - Stanford CS230 Deep Learning notes https://cs230-stanford.github.io

# %% [markdown]
# # Example Network
# Just to get an idea of how PyTorch feels like here are some examples of networks.

# %%
from collections import OrderedDict

import torch                     # basic tensor functions
import torch.nn as nn            # everything neural network
import torch.nn.functional as F  # functional/stateless version of nn
import torch.optim as optim      # optimizers :)

# %%
# Simple sequential model
model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5),
    nn.ReLU(),
    nn.Conv2d(20, 64, 5),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
)

# %%
model

# %%
# forward pass
model(torch.rand(16, 1, 32, 32)).shape

# %%
# Simple sequential model with named layers
layers = OrderedDict([
    ("conv1", nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5)),
    ("relu1", nn.ReLU()),
    ("conv2", nn.Conv2d(20,64,5)),
    ("relu2", nn.ReLU()),
    ("aavgp", nn.AdaptiveAvgPool2d(1)),
])
model = nn.Sequential(layers)
model


# %%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.adaptive_avg_pool2d(x, 1)
        return x


model = Net()
model

# %% [markdown]
# # Versions

# %%
import torch
torch.__version__

# %%
import torchvision
torchvision.__version__

# %%
import numpy as np
np.__version__
