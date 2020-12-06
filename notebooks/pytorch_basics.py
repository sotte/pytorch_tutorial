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

# %% [markdown] toc-hr-collapsed=true toc-nb-collapsed=true
# # PyTorch Basics
# - tensors like numpy
# - tensors on the gpu
# - tensors and automatic derivatives
# - tensors as neural network abstractions: `torch.nn`
# - optimizers: `nn.optim`

# %% [markdown]
# ## Init, helpers, utils, ...

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

# %% [markdown] toc-hr-collapsed=true toc-nb-collapsed=true
# # Tensors
# tensors - the atoms of machine learning

# %% [markdown]
# ## Tensors in numpy and pytorch

# %%
import numpy as np
from numpy.linalg import inv
from numpy.linalg import multi_dot as mdot

# %%
# numpy
np.eye(3)

# %%
# torch
torch.eye(3)

# %%
# numpy
X = np.random.random((5, 3))
X

# %%
# pytorch
Y = torch.rand((5, 3))
Y

# %%
X.shape

# %%
Y.shape

# %%
# numpy
X.T @ X

# %%
# torch
Y.t() @ Y

# %%
# numpy
inv(X.T @ X)

# %%
# torch
torch.inverse(Y.t() @ Y)

# %% [markdown]
# ## More on PyTorch Tensors

# %% [markdown]
# Operations are also available as methods.

# %%
A = torch.eye(3)
A.add(1)

# %%
A

# %% [markdown]
# Any operation that mutates a tensor in-place has a `_` suffix.

# %%
A.add_(1)
A

# %% [markdown]
# ## Indexing and broadcasting
# It works as expected/like numpy:

# %%
A[0, 0]

# %%
A[0]

# %%
A[0:2]

# %%
A[:, 1:3]

# %% [markdown]
# ## Converting

# %%
A = torch.eye(3)
A

# %%
# torch --> numpy
B = A.numpy()
B

# %% [markdown]
# Note: torch and numpy can share the same memory / zero-copy

# %%
A.add_(.5)
A

# %%
B

# %%
# numpy --> torch
torch.from_numpy(np.eye(3))

# %% [markdown]
# ## Much more

# %%
[o for o in dir(torch) if not o.startswith("_")]

# %%
[o for o in dir(A) if not o.startswith("_")]

# %% [markdown]
# # But what about the GPU?
# How do I use the GPU?
#
# If you have a GPU make sure that the right pytorch is installed
# (check https://pytorch.org/ for details).

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

# %% [markdown]
# If you have a GPU you should get something like: 
# `device(type='cuda', index=0)`
#
# You can move data to the GPU by doing `.to(device)`.

# %%
data = torch.eye(3)
data = data.to(device)
data

# %% [markdown]
# Now the computation happens on the GPU.

# %%
res = data + data
res

# %%
res.device

# %% [markdown]
# Note: before `v0.4` one had to use `.cuda()` and `.cpu()` to move stuff to and from the GPU.
# This littered the code with many:
# ```python
# if CUDA:
#     model = model.cuda()
# ```

# %% [markdown]
# # Automatic differentiation with `autograd`
# Prior to `v0.4` PyTorch used the class `Variable` to record gradients. You had to wrap `Tensor`s in `Variable`s.
# `Variable`s behaved exactly like `Tensors`.
#
# With `v0.4` `Tensor` can record gradients directly if you tell it do do so, e.g. `torch.ones(3, requires_grad=True)`.
# There is no need for `Variable` anymore.
# Many tutorials still use `Variable`, be aware!
#
# Ref:
# - https://pytorch.org/docs/stable/autograd.html
# - https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

# %% [markdown]
# You rarely use `torch.autograd` directly.
# Pretty much everything is part or `torch.Tensor` now.
# Simply add `requires_grad=True` to the tensors you want to calculate the gradients for.
# `nn.Module` track gradients automatically.

# %%
from torch import autograd

# %%
x = torch.tensor(2.)
x

# %%
x = torch.tensor(2., requires_grad=True)
x

# %%
print(x.requires_grad)

# %%
print(x.grad)

# %%
y = x ** 2

print("Grad of x:", x.grad)

# %%
y = x ** 2
y.backward()

print("Grad of x:", x.grad)

# %%
# What is going to happen here?
# x = torch.tensor(2.)
# x.backward()

# %%
# Don't record the gradient
# Useful for inference

params = torch.tensor(2., requires_grad=True)

with torch.no_grad():
    y = x * x
    print(x.grad_fn)

# %% [markdown]
# `nn.Module` and `nn.Parameter` keep track of gradients for you.

# %%
lin = nn.Linear(2, 1, bias=True)
lin.weight

# %%
type(lin.weight)

# %%
isinstance(lin.weight, torch.FloatTensor)

# %% [markdown]
# ## `torch.nn`
# The neural network modules contains many different layers.

# %%
from torch import nn

# %%
lin_reg = nn.Linear(1, 1, bias=True)
lin_reg

# %%
nn.Conv2d

# %%
nn.Conv3d

# %%
nn.BatchNorm2d

# %% [markdown]
# ### Activations

# %%
nn.ReLU

# %%
nn.Sigmoid

# %% [markdown]
# ### Losses

# %%
nn.Softmax

# %%
nn.CrossEntropyLoss

# %%
nn.BCELoss

# %%
nn.MSELoss

# %% [markdown]
# ### Functional (stateless) alternatives

# %%
from torch.nn import functional as F

# %%
F.mse_loss

# %%
F.relu

# %%
F.relu6

# %% [markdown]
# ## `torch.optim`

# %%
from torch import optim

# %%
optim.SGD

# %%
optim.Adam

# %%
optim.AdamW

# %% [markdown]
# # Exercise
# - Do you remember the analytical solution to solve for the parameters of linear regression? Implement it.
