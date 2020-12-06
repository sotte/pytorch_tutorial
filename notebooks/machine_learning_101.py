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
# # ML 101 Recap
#
# **ML = model + loss + optimizer**
#
#
# ## Linear regression example
#
# 0. Data
#
# 1. Model:
#   - $f(X) = X \beta = \hat y$
#
# 2. Loss / criterion:
#   - $ err_i = y_i - f(X_i)$
#   - $MSE = \frac{1}{n} \sum_{i=1}^{N} err_i^2$
#   
# 3. Optimize:
#   - minimize the MSE yields the optimal $\hat\beta$ (after doing some math)
#   - $\hat\beta = (X^TX)^{-1}X^Ty$
#   - (or, more generally, use gradient descent to optimize the parameters)

# %%
import numpy as np
from numpy.linalg import inv
from numpy.linalg import multi_dot as mdot

import matplotlib.pyplot as plt

# %matplotlib inline

# %% [markdown]
# ## LinReg with numpy

# %%
X = np.random.random((5, 3))
y = np.random.random(5)
X.shape, y.shape

# %% [markdown]
# Calculate the optimal parameter:
# $\hat\beta = (X^T X)^{-1} X^T y$

# %%
XT = X.T  # transpose

beta_ = mdot([inv(XT @ X), XT, y])
beta_

# %%
XT = X.T  # transpose

beta_ = inv(XT @ X) @ XT @ y
beta_


# %% [markdown]
# The model $f$:

# %%
def f(X, beta):
    return X @ beta

f(X, beta_)

# %% [markdown]
# ## LinReg with PyTorch

# %%
import torch

# %%
# X = torch.rand((5, 3))
# y = torch.rand(5)
X = torch.from_numpy(X)
y = torch.from_numpy(y)
X.shape, y.shape

# %% [markdown]
# $\hat\beta = (X^T X)^{-1} X^T y$

# %%
XT = X.t()

beta__ = (XT @ X).inverse() @ XT @ y
beta__

# %%
beta__.numpy() - beta_

# %% [markdown]
# ## LinReg with PyTorch and Gradent Descent
#
# Previously, we had to do some math to calculate the optimal $\hat\beta$.
# PyTorch calculates the gradients for us automatically (more on that later)
# and we can use some version of gradient desctent to find our $\hat\beta$.

# %%
from sklearn.datasets import make_regression

n_features = 1
n_samples = 100

X, y = make_regression(
    n_samples=n_samples,
    n_features=n_features,
    noise=10,
)

dom_np = np.linspace(X.min(), X.max(), 20)
dom = torch.from_numpy(dom_np).unsqueeze(-1).float()

fix, ax = plt.subplots()
ax.plot(X, y, ".")

# %%
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float().unsqueeze(-1)
X.shape, y.shape

# %%
from torch import nn

class LinReg(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.beta = nn.Linear(input_dim, 1)
        
    def forward(self, X):
        return self.beta(X)


model = LinReg(n_features)

# %%
loss_fn = nn.MSELoss()

# %%
from torch import optim

optimizer = optim.SGD(model.parameters(), lr=0.01)

# %%
# Train step
model.train()
optimizer.zero_grad()

y_ = model(X)

loss = loss_fn(y_, y)
loss.backward()
optimizer.step()

# Eval
model.eval()
with torch.no_grad():
    y_ = model(dom)
    

# Vis
fig, ax = plt.subplots()
ax.plot(X.numpy(), y.numpy(), ".", label="data")
ax.plot(dom_np, y_.numpy(), "-", label="pred")
ax.set_title(f"MSE: {loss.item():0.1f}")
ax.legend();

# %%
model.beta

# %%
model.beta.weight

# %%
model.beta.weight.data

# %%
model.beta.bias

# %% [markdown]
# ## LinReg with GPU
#
# Simply move the data and the model to the GPU.

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = LinReg(n_features).to(device)  # <-- here
optimizer = optim.SGD(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

X, y = X.to(device), y.to(device)  # <-- here
dom = dom.to(device)

# %% [markdown]
# The rest stays the same.

# %%
# Train step
model.train()
optimizer.zero_grad()

y_ = model(X)

loss = loss_fn(y_, y)
loss.backward()
optimizer.step()

# Eval
model.eval()
with torch.no_grad():
    y_ = model(dom)
    

# Vis
fig, ax = plt.subplots()
ax.plot(X.cpu().numpy(), y.cpu().numpy(), ".", label="data")
ax.plot(dom_np, y_.cpu().numpy(), "-", label="pred")
ax.set_title(f"MSE: {loss.cpu().item():0.1f}")
ax.legend();
