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
# # Clustering with PyTorch

# %% [markdown]
# "PyTorch is a python package that provides [...]
# Tensor computation (like numpy) with strong GPU acceleration [...]"
#
# So, let's use it for some Mean-shift clustering.

# %%
import math
import operator

import numpy as np
import matplotlib.pyplot as plt

import torch

# %matplotlib inline

# %% [markdown]
# # Mean shitft clustering with numpy

# %% [markdown]
# ## Create data

# %%
n_clusters = 6
n_samples = 1000

# %% [markdown]
# To generate our data, we're going to pick `n_clusters` random points, which we'll call centroids, and for each point we're going to generate `n_samples` random points about it.

# %%
centroids = np.random.uniform(-35, 35, (n_clusters, 2))
slices = [np.random.multivariate_normal(centroids[i], np.diag([5., 5.]), n_samples)
          for i in range(n_clusters)]
data = np.concatenate(slices).astype(np.float32)


# %% [markdown]
# Plot the data and the centroids:

# %%
def plot_data(centroids, data, n_samples):
    colour = plt.cm.rainbow(np.linspace(0,1,len(centroids)))

    fig, ax = plt.subplots(figsize=(4, 4))
    for i, centroid in enumerate(centroids):
        samples = data[i * n_samples : (i + 1) * n_samples]
        ax.scatter(samples[:, 0], samples[:, 1], c=colour[i], s=1)
        ax.plot(centroid[0], centroid[1], markersize=10, marker="x", color='k', mew=5)
        ax.plot(centroid[0], centroid[1], markersize=5, marker="x", color='m', mew=2)
    plt.axis('equal')
    
plot_data(centroids, data, n_samples)

# %% [markdown]
# ## The mean shift algorithm
#
# "Mean shift is a **non-parametric** feature-space analysis technique for locating the maxima of a density function, a so-called **mode-seeking algorithm**. Application domains include cluster analysis in computer vision and image processing." -- https://en.wikipedia.org/wiki/Mean_shift
#
# Think of mean-shift clustering as k-means but you don't have to specify the number of clusters.
# (You have to specify the **bandwidth** but that can be automated.)

# %% [markdown]
# Algo:
# ```python
# # PSEUDO CODE
# while not_converged():
#     for i, point in enumerate(points):
#         # distance for the given point to all other points
#         distances = calc_distances(point, points)
#         
#         # turn distance into weights using a gaussian
#         weights = gaussian(dist, bandwidth=2.5)
#         
#         # update the weights by using the weights
#         points[i] = (weights * points).sum(0) / weights.sum()
#
# return points
# ```

# %% [markdown]
# ## The implementation
#
# Let's implement this with numpy:

# %%
from numpy import exp, sqrt, array


# %%
def distance(x, X):
    # return np.linalg.norm(x - X, axis=1)
    return sqrt(((x - X)**2).sum(1))


# %% [markdown]
# Let's try it out. (More on how this function works shortly)

# %%
a = array([1, 2])
b = array([[1, 2],
           [2, 3],
           [-1, -3]])

dist = distance(a, b)
dist


# %%
def gaussian(dist, bandwidth):
    return exp(-0.5 * ((dist / bandwidth))**2) / (bandwidth * math.sqrt(2 * math.pi))


# %%
gaussian(dist, 2.5)


# %% [markdown]
# Now we can do a single mean shift step:

# %%
def meanshift_step(X, bandwidth=2.5):
    for i, x in enumerate(X):
        dist = distance(x, X)
        weight = gaussian(dist, bandwidth)
        X[i] = (weight[:, None] * X).sum(0) / weight.sum()
    return X


# %% [markdown]
# Data before:

# %%
plot_data(centroids, data, n_samples)

# %% [markdown]
# Data after:

# %%
_X = meanshift_step(np.copy(data))
plot_data(centroids, _X, n_samples)


# %% [markdown]
# Just repeath this/iterate a few times and we have the complete mean shift algorithm:

# %%
def meanshift(X):
    X = np.copy(X)
    for _ in range(5):
        X = meanshift_step(X)
    return X


# %%
# %%time
X = meanshift(data)

# %%
plot_data(centroids, X, n_samples)

# %% [markdown]
# # Mean shift in PyTorch (with GPU)
#
# PyTorch is like numpy and the interface is very similar.
#
# We actually don't have to adjust anything really to use torch instead of numpy.

# %%
import torch
from torch import exp, sqrt


# %% [markdown]
# We oncly have to copy the data into a PyTorch GPU tensor.

# %%
def meanshift_torch(X):
    X = torch.from_numpy(np.copy(X)).cuda()
    for it in range(5):
        X = meanshift_step(X)
    return X


# %%
# %time X = meanshift_torch(data).cpu().numpy()
plot_data(centroids+2, X, n_samples)


# %% [markdown]
# Same results, but the implementation is about the same speed.
#
# CUDA kernels have to be started for each calculation and the kernels don't have enough to do.
# Let's not process individual points, but batches of points.

# %% [markdown]
# ## Batch processing

# %%
def distance_batch(a, b):
    return sqrt(((a[None,:] - b[:,None]) ** 2).sum(2))


# %%
a = torch.rand(2, 2)
b = torch.rand(3, 2)
distance_batch(b, a)


# %% [markdown]
# `distance_batch` contains some broadcast magic that allows us to compute the distance from each point in a batch to all points in the data.

# %%
def meanshift_torch2(data, batch_size=500):
    n = len(data)
    X = torch.from_numpy(np.copy(data)).cuda()
    for _ in range(5):
        for i in range(0, n, batch_size):
            s = slice(i, min(n, i + batch_size))
            weight = gaussian(distance_batch(X, X[s]), 2.5)
            num = (weight[:, :, None] * X).sum(dim=1)
            X[s] = num / weight.sum(1)[:, None]
    return X


# %%
# %time X = meanshift_torch2(data, batch_size=1).cpu().numpy()

# %%
# %time X = meanshift_torch2(data, batch_size=10).cpu().numpy()

# %%
# %time X = meanshift_torch2(data, batch_size=100).cpu().numpy()

# %%
# %time X = meanshift_torch2(data, batch_size=1000).cpu().numpy()

# %%
# %time X = meanshift_torch2(data, batch_size=6000).cpu().numpy()

# %%
plot_data(centroids+2, X, n_samples)

# %% [markdown]
# # Mean shift in scikit-learn
#
# Of course, sklearn also offers `MeanShift`.
# Let's see how it performs

# %%
from sklearn.cluster import MeanShift

# %%
# %%time
model = MeanShift()
model.fit(data)

# %% [markdown]
# This is a faster than our naive implementation, but much slower than the GPU version.
#

# %% [markdown]
# # Note
# Keep in mind that this demo is not saying that A is faster than B.
# It rather shows that you can use PyTorch in fun ways!
#
# Ref:
# - https://pytorch.org/docs/stable/notes/broadcasting.html
# - https://pytorch.org/docs/stable/notes/cuda.html
# - https://github.com/fastai/fastai/blob/master/tutorials/meanshift.ipynb
