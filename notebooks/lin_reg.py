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
# # LinReg with PyTorch, Gradient Descent, and GPU

# %% [markdown]
# ## Init, helpers, utils ...

# %%
# %matplotlib inline

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE

# %%
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from IPython.core.debugger import set_trace

# %% [markdown]
# # The Problem

# %%
from sklearn.datasets import make_regression


n_features = 1
n_samples = 100

X, y = make_regression(
    n_samples=n_samples,
    n_features=n_features,
    noise=20,
    random_state=42,
)

fix, ax = plt.subplots()
ax.plot(X, y, ".")

# %% [markdown]
# # The Solution

# %%
X = torch.from_numpy(X).float()
y = torch.from_numpy(y.reshape((n_samples, n_features))).float()


# %%
class LinReg(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.beta = nn.Linear(input_dim, 1)
        
    def forward(self, X):
        return self.beta(X)

# or just
# model = nn.Linear(input_dim, 1)


# %%
model = LinReg(n_features).to(DEVICE)  # <-- here
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


X, y = X.to(DEVICE), y.to(DEVICE)  # <-- here

# %%
# Train step
model.train()  # <-- here
optimizer.zero_grad()

y_ = model(X)
loss = loss_fn(y_, y)

loss.backward()
optimizer.step()

# Eval
model.eval()  # <-- here
with torch.no_grad():
    y_ = model(X)    

# Vis
fig, ax = plt.subplots()
ax.plot(X.cpu().numpy(), y_.cpu().numpy(), ".", label="pred")
ax.plot(X.cpu().numpy(), y.cpu().numpy(), ".", label="data")
ax.set_title(f"MSE: {loss.item():0.1f}")
ax.legend();

# %% [markdown]
# Note: I did gradient descent with all the data. I did not split the data into `train` and `valid` which should be done!

# %%
list(range(100))

# %% [markdown]
# # Exercise:
# - Write a proper training loop for this linear regression example.
# - Split data into train and valid.
# - Use the Dataset and DataLoader abstraction.
# - Create a logistic regression module.
# - Create a Multi Layer Perceptron (MLP).
