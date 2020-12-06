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
# # `torch.jit`
#
# Eager execution is great for development and debugging. but it can be hard to (automatically) optimize the code and deploy it.
#
# Now there is`torch.jit` with two flavours:
#
# - `torch.jit.trace` does not record control flow.
# - `torch.jit.script` records control flow and creates an intermediate representation that can be optimized; only supports a subset of Python.
#
# Note: don't forget `model.eval()` and `model.train()`.
#
#
# ## Ref and More:
# - https://pytorch.org/docs/stable/jit.html
# - https://speakerdeck.com/perone/pytorch-under-the-hood
# - https://lernapparat.de/fast-lstm-pytorch/

# %% [markdown]
# ## Init, helpers, utils, ...

# %%
# %load_ext autoreload
# %autoreload 2

# %matplotlib inline

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

# %%
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from IPython.core.debugger import set_trace

import utils  # little helpers
from utils import attr


# %% [markdown]
# # `torch.jit.trace`

# %%
def f(x):
    if x.item() < 0:
        return torch.tensor(0)
    else:
        return x


# %%
f(torch.tensor(-1))

# %%
f(torch.tensor(3))

# %%
X = torch.tensor(1)
traced = torch.jit.trace(f, X)

# %%
type(traced)

# %%
traced(torch.tensor(1))

# %%
traced.graph

# %%
traced(torch.tensor(-1))

# %% [markdown]
# ## Storing and restoring

# %%
traced.save("traced.pt")

# %%
# !file scripted.pt

# %%
g = torch.jit.load("traced.pt")

# %%
g(torch.tensor(1))

# %%
g(torch.tensor(-1))

# %% [markdown]
# # `torch.jit.script`

# %%
bool(torch.tensor(1) < 2)


# %%
@torch.jit.script
def f(x):
    if bool(x < 0):
        result = torch.zeros(1)
    else:
        result = x
    return result


# %% [markdown]
# This is `torchscript` which is a only a supset of python.

# %%
f(torch.tensor(-1))

# %%
f(torch.tensor(1))

# %%
type(f)

# %%
f.graph

# %% [markdown]
# ## Storing and restoring

# %%
torch.jit.save(f, "scripted.pt")

# %%
# !file scripted.pt

# %%
g = torch.jit.load("scripted.pt")

# %%
g(torch.tensor(-1))

# %%
g(torch.tensor(1))

# %% [markdown]
# ## Subclassing `torch.jit.ScriptModule`
# If you work with `nn.Module` replace it by `torch.jit.ScriptModule` (see [[tutorial]](https://pytorch.org/tutorials/beginner/deploy_seq2seq_hybrid_frontend_tutorial.html) for more).
#
# ```python
# class MyModule(torch.jit.ScriptModule):
#     def __init__(self):
#         super().__init__()
#         
#     def forward(self, x):
#         # ...
#         return x
# ```

# %% [markdown]
# # PyTorch and C++
#
# PyTorch offers a very nice(!) C++ interface which is very close to Python.

# %% [markdown]
# ## Loading traced models from C++

# %% [markdown]
# ```c++
# #include <torch/script.h>
#
# int main(int(argc, const char* argv[]) {
#     auto module = torch::jit::load("scrpted.pt");
#     // data ...
#     module->forward(data);
# }
# ```
