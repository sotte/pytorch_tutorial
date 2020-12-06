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
# # Transfer learning with PyTorch
# We're going to train a neural network to classify dogs and cats.

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
DEVICE

# %%
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from IPython.core.debugger import set_trace


# %%
# # %load my_train_helper.py
def get_trainable(model_params):
    return (p for p in model_params if p.requires_grad)


def get_frozen(model_params):
    return (p for p in model_params if not p.requires_grad)


def all_trainable(model_params):
    return all(p.requires_grad for p in model_params)


def all_frozen(model_params):
    return all(not p.requires_grad for p in model_params)


def freeze_all(model_params):
    for param in model_params:
        param.requires_grad = False



# %% [markdown] toc-hr-collapsed=true
# # The Data - DogsCatsDataset

# %% [markdown]
# ## Transforms

# %%
from torchvision import transforms

_image_size = 224
_mean = [0.485, 0.456, 0.406]
_std = [0.229, 0.224, 0.225]


train_trans = transforms.Compose([
    transforms.Resize(256),  # some images are pretty small
    transforms.RandomCrop(_image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(.3, .3, .3),
    transforms.ToTensor(),
    transforms.Normalize(_mean, _std),
])
val_trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(_image_size),
    transforms.ToTensor(),
    transforms.Normalize(_mean, _std),
])

# %% [markdown]
# ## Dataset
#
# The implementation of the dataset does not really.

# %%
from torchvision.datasets.folder import ImageFolder

# %%
train_ds = ImageFolder("dogscats/training_set/", transform=train_trans)
val_ds = ImageFolder("dogscats/test_set/", transform=val_trans)

batch_size = 32
n_classes = 2

# %% [markdown]
# Use the following if you want to use the full dataset:

# %%
# train_ds = DogsCatsDataset("../data/raw", "train", transform=train_trans)
# val_ds = DogsCatsDataset("../data/raw", "valid", transform=val_trans)

# batch_size = 128
# n_classes = 2

# %%
len(train_ds), len(val_ds)

# %% [markdown]
# ## DataLoader
# Batch loading for datasets with multi-processing and different sample strategies.

# %%
from torch.utils.data import DataLoader


train_dl = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=16,
)

val_dl = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=16,
)

# %% [markdown]
# # The Model
# PyTorch offers quite a few [pre-trained networks](https://pytorch.org/docs/stable/torchvision/models.html) such as:
# - AlexNet
# - VGG
# - ResNet
# - SqueezeNet
# - DenseNet
# - Inception v3
#
# And there are more available via [pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch):
# - NASNet,
# - ResNeXt,
# - InceptionV4,
# - InceptionResnetV2, 
# - Xception, 
# - DPN,
# - ...
#
# We'll use a simple resnet18 model:

# %%
from torchvision import models

model = models.resnet18(pretrained=True)

# %%
model

# %%
import torchsummary

torchsummary.summary(model, (3, 224, 224), device="cpu")

# %%
nn.Linear(2, 1, bias=True)

# %%
# Freeze all parameters manually
for param in model.parameters():
    param.requires_grad = False

# %%
# Or use our convenient functions from before
freeze_all(model.parameters())
assert all_frozen(model.parameters())

# %% [markdown]
# Replace the last layer with a linear layer. New layers have `requires_grad = True`.

# %%
model.fc = nn.Linear(512, n_classes)

# %%
assert not all_frozen(model.parameters())


# %%
def get_model(n_classes=2):
    model = models.resnet18(pretrained=True)
    freeze_all(model.parameters())
    model.fc = nn.Linear(512, n_classes)
    model = model.to(DEVICE)
    return model


model = get_model()

# %% [markdown]
# # The Loss

# %%
criterion = nn.CrossEntropyLoss()

# %% [markdown]
# # The Optimizer

# %%
optimizer = torch.optim.Adam(
    get_trainable(model.parameters()),
    lr=0.001,
    # momentum=0.9,
)

# %% [markdown]
# # The Train Loop

# %%
N_EPOCHS = 1

for epoch in range(N_EPOCHS):
    
    # Train
    model.train()  # IMPORTANT
    
    total_loss, n_correct, n_samples = 0.0, 0, 0
    for batch_i, (X, y) in enumerate(train_dl):
        X, y = X.to(DEVICE), y.to(DEVICE)
        
        optimizer.zero_grad()
        y_ = model(X)
        loss = criterion(y_, y)
        loss.backward()
        optimizer.step()
        
        # Statistics
        print(
            f"Epoch {epoch+1}/{N_EPOCHS} |"
            f"  batch: {batch_i} |"
            f"  batch loss:   {loss.item():0.3f}"
        )
        _, y_label_ = torch.max(y_, 1)
        n_correct += (y_label_ == y).sum().item()
        total_loss += loss.item() * X.shape[0]
        n_samples += X.shape[0]
    
    print(
        f"Epoch {epoch+1}/{N_EPOCHS} |"
        f"  train loss: {total_loss / n_samples:9.3f} |"
        f"  train acc:  {n_correct / n_samples * 100:9.3f}%"
    )
    
    
    # Eval
    model.eval()  # IMPORTANT
    
    total_loss, n_correct, n_samples = 0.0, 0, 0
    with torch.no_grad():  # IMPORTANT
        for X, y in val_dl:
            X, y = X.to(DEVICE), y.to(DEVICE)
                    
            y_ = model(X)
        
            # Statistics
            _, y_label_ = torch.max(y_, 1)
            n_correct += (y_label_ == y).sum().item()
            loss = criterion(y_, y)
            total_loss += loss.item() * X.shape[0]
            n_samples += X.shape[0]

    
    print(
        f"Epoch {epoch+1}/{N_EPOCHS} |"
        f"  valid loss: {total_loss / n_samples:9.3f} |"
        f"  valid acc:  {n_correct / n_samples * 100:9.3f}%"
    )


# %% [markdown]
# # Exercise
# - Create your own module which takes any of the existing pre-trained model as backbone and adds a problem specific head.

# %%
class Net(nn.Module):
    def __init__(self, backbone: nn.Module, n_classes: int):
        super().__init__()
        # self.backbone
        # self.head = init_head(n_classes)
        
    def forward(self, x):
        # TODO
        return x
