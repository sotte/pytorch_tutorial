# -*- coding: utf-8 -*-
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
# # Working with Data: `Dataset`, `DataLoader`, `Sampler`, and `Transforms`
#
# These basic concepts make it easy to work with large data.

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
# # Dataset
# It's easy to create your `Dataset`,
# but PyTorch comes with some
# [build-in datasets](https://pytorch.org/docs/stable/torchvision/datasets.html):
#
# - MNIST
# - Fashion-MNIST
# - KMNIST
# - EMNIST
# - FakeData
# - COCO
#   - Captions
#   - Detection
# - LSUN
# - ImageFolder
# - DatasetFolder
# - Imagenet-12
# - CIFAR
# - STL10
# - SVHN
# - PhotoTour
# - SBU
# - Flickr
# - VOC
# - Cityscapes
#
# `Dataset` gives you information about the number of samples (implement `__len__`) and gives you the sample at a given index (implement `__getitem__`.
# It's a nice and simple abstraction to work with data.

# %%
from torch.utils.data import Dataset

# %% [markdown]
# ```python
# class Dataset(object):
#     def __getitem__(self, index):
#         raise NotImplementedError
#
#     def __len__(self):
#         raise NotImplementedError
#
#     def __add__(self, other):
#         return ConcatDataset([self, other])
# ```

# %% [markdown]
# The `ImageFolder` dataset is quite useful and follows the usual conventions for folder layouts:
#
# ```
# root/dog/xxx.png
# root/dog/xxy.png
# root/dog/xxz.png
#
# root/cat/123.png
# root/cat/nsdf3.png
# root/cat/asd932_.png
# ```

# %% [markdown]
# ## Example: dogs and cats dataset
# Please download the dataset from
# https://www.kaggle.com/chetankv/dogs-cats-images
# and place it in the `notebook/` folder.

# %%
# !tree -d dogscats/

# %%
from torchvision.datasets.folder import ImageFolder

train_ds = ImageFolder("dogscats/training_set/")

# %%
train_ds

# %%
# the __len__ method
len(train_ds)

# %%
# the __getitem__ method
train_ds[0]

# %%
train_ds[0][0]

# %%
train_ds[0][1]

# %% [markdown]
# Optionally, some datasets offer convenience functions and attributes.
# This is not enforced by the interface! Don't rely on it!

# %%
train_ds.classes

# %%
train_ds.class_to_idx

# %%
train_ds.imgs

# %%

# %%
import random

rand_idx = np.random.randint(0, len(train_ds), 4)
for i in rand_idx:
    img, label_id = train_ds[i]
    print(label_id, train_ds.classes[label_id], i)
    display(img)

# %% [markdown]
# # `torchvision.transforms`
#
# Common image transformation that can be composed/chained [[docs]](https://pytorch.org/docs/stable/torchvision/transforms.html).

# %%
from torchvision import transforms

# %%
_image_size = 224
_mean = [0.485, 0.456, 0.406]
_std = [0.229, 0.224, 0.225]


trans = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(_image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(.3, .3, .3),
    transforms.ToTensor(),
    transforms.Normalize(_mean, _std),
])

trans(train_ds[7074][0])

# %% [markdown]
# ## `torchvision.transforms.functional`
#
# >Functional transforms give you fine-grained control of the transformation pipeline. As opposed to the transformations above, functional transforms don’t contain a random number generator for their parameters. That means you have to specify/generate all parameters, but you can reuse the functional transform. For example, you can apply a functional transform to multiple images like this:
# >
# > https://pytorch.org/docs/stable/torchvision/transforms.html#functional-transforms
#
# ```python
# import torchvision.transforms.functional as TF
# import random
#
# def my_segmentation_transforms(image, segmentation):
#     if random.random() > 5:
#         angle = random.randint(-30, 30)
#         image = TF.rotate(image, angle)
#         segmentation = TF.rotate(segmentation, angle)
#     # more transforms ...
#     return image, segmentation
# ```

# %% [markdown]
# Ref:
# - https://pytorch.org/docs/stable/torchvision/transforms.htm
# - https://pytorch.org/docs/stable/torchvision/transforms.html#functional-transforms
# - https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# - https://github.com/mdbloice/Augmentor
# - https://github.com/aleju/imgaug
#
# Shout-out:
# - Hig performance image augmentation with pillow-simd [[github]](https://github.com/uploadcare/pillow-simd) [[benchmark]](http://python-pillow.org/pillow-perf/)
# - Improving Deep Learning Performance with AutoAugment [[blog]](https://ai.googleblog.com/2018/06/improving-deep-learning-performance.html) [[paper]](https://arxiv.org/abs/1805.09501) [[pytorch implementation]](https://github.com/DeepVoltaire/AutoAugment)

# %% [markdown]
# # Dataloader
# The `DataLoader` class offers batch loading of datasets with multi-processing and different sample strategies [[docs]](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).
#
# The signature looks something like this:
# ```python
# DataLoader(
#     dataset,
#     batch_size=1,
#     shuffle=False,
#     sampler=None,
#     batch_sampler=None,
#     num_workers=0,
#     collate_fn=default_collate,
#     pin_memory=False,
#     drop_last=False,
#     timeout=0,
#     worker_init_fn=None
# )
# ```

# %%
from torch.utils.data import DataLoader

# %%
train_ds = ImageFolder("dogscats/training_set/", transform=trans)
train_dl = DataLoader(
    train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=4,
)

# %%
train_iter = iter(train_dl)
X, y = next(train_iter)

# %%
print("X:", X.shape)
print("y:", y.shape)

# %% [markdown]
# Note that I passed `trans`, which returns `torch.Tensor`, not pillow images.
# DataLoader expects tensors, numbers, dicts or lists.

# %%
_train_ds = ImageFolder("dogscats/test_set/", transform=trans) 
_train_dl = DataLoader(_train_ds, batch_size=2, shuffle=True)


# %% [markdown]
# ## `collate_fn`
# The `collate_fn` argument of `DataLoader` allows you to customize how single datapoints are put together into a batch.
# `collate_fn` is a simple callable that gets a list of datapoints (i.e. what `dataset.__getitem__` returns).

# %% [markdown]
# Example of a custom `collate_fn`
# (taken from [here](https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/3)):

# %%
def my_collate_fn(list_of_x_y):
    data = [item[0] for item in list_of_x_y]
    target = [item[1] for item in list_of_x_y]
    target = torch.LongTensor(target)
    return [data, target]


# %% [markdown]
# # Sampler
# `Sampler` define **how** to sample from the dataset [[docs]](https://pytorch.org/docs/stable/data.html#torch.utils.data.sampler.Sampler).
#
# Examples:
# - `SequentialSampler`
# - `RandomSamples`
# - `SubsetSampler`
# - `WeightedRandomSampler`
#
# Write your own by simply implementing `__iter__` to iterate over the indices of the dataset.
#
# ```python
# class Sampler(object):
#     def __init__(self, data_source):
#         pass
#
#     def __iter__(self):
#         raise NotImplementedError
#
#     def __len__(self):
#         raise NotImplementedError
# ```

# %% [markdown]
# # Recap
# - `Dataset`: get one datapoint
# - `transforms`: composable transformations
# - `DataLoader`: combine single datapoints into batches (plus multi processing and more)
# - `Sampler`: **how** to sample from a dataset
#
# **Simple but extensible interfaces**

# %% [markdown]
# # Exercise
# Go out and play:
#
# - Maybe extend the `DogsCatsDataset` such that you can specify the size of dataset, i.e. the number of samples.
# - Maybe try the `Subset` [[docs]](https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset) to create smaller datasets.
# - Maybe create `SubsetFraction` where you can specify the size of the dataset (between 0. and 1.).
# - Maybe write a custom collate function for the `DogsCatsDataset` that turns it into a dataset appropriate to use in an autoencoder settings.

# %%
def autoencoder_collate_fn(list_of_x_y):
    # TODO implement me
    pass


# %%
class MyDataSet(Dataset):
    def __init__(self):
        super().__init__()
        # TODO implement me
    
    def __len__(self):
        # TODO implement me
        pass
    
    def __getitem__(self, idx):
        # TODO implement me
        pass
