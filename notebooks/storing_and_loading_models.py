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
# # Storing and Loading Models
#
# https://pytorch.org/tutorials/beginner/saving_loading_models.html

# %% [markdown]
# ## Init, helpers, utils, ...

# %%
# %matplotlib inline

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from IPython.core.debugger import set_trace


# %% [markdown]
# # `state_dict()`

# %% [markdown]
# ## `nn.Module.state_dict()`
# `nn.Module` contain state dict, that maps each layer to the learnable parameters.

# %%
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# %%
model = Net()

# %%
model.state_dict()


# %%
def state_dict_info(obj):
    print(f"{'layer':25} shape")
    print("===================================================")
    for k,v in obj.state_dict().items():
        try:
            print(f"{k:25} {v.shape}")
        except AttributeError:
            print(f"{k:25} {v}")


# %%
state_dict_info(model)

# %% [markdown]
# ## `nn.Optimizer`
#
# Optimizers also have a a `state_dict`.

# %%
optimizer = optim.Adadelta(model.parameters())

# %%
state_dict_info(optimizer)

# %%
optimizer.state_dict()["state"]

# %%
optimizer.state_dict()["param_groups"]

# %% [markdown]
# ## Storing and loading `state_dict`

# %%
model_file = "model_state_dict.pt"
torch.save(model.state_dict(), model_file)

# %%
model = Net()
model.load_state_dict(torch.load(model_file))

# %% [markdown]
# ## Storing and loading the full model

# %%
model_file = "model_123.pt"
torch.save(model, model_file)

# %%
# Only works if code for `Net` is available right now
model = torch.load(model_file)

# %% [markdown]
# # Example Checkpointing
# You can store model, optimizer and arbitrary information and reload it.
#
# Example:
# ```python
# torch.save(
#     {
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'epoch': epoch,
#         'loss': loss,
#     },
#     PATH,
# )
# ```

# %% [markdown]
# # Exercise
# - Find out what is going to be in the `state` variable of the `state_dict` of an optimizer.
# - Write your own checkpoint functionality.
