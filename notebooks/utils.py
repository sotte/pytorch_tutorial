import os
import zipfile

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models
from torchvision.datasets.folder import ImageFolder, default_loader
from torchvision.datasets.utils import download_url, check_integrity
from my_datasets import DogsCatsDataset


################################################################################
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


################################################################################
# Helpers
def attr(obj):
    """
    Return all public attributes of an object.
    """
    return [x for x in dir(obj) if not x.startswith("_")]


def ptitle(title):
    print("#" * 80)
    print(f"# {title}")



################################################################################
# DOGS AND CATS DEMO

def get_model(n_classes=2):
    model = models.resnet18(pretrained=True)
    # Freeze all
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(512, n_classes)
    model = model.to(DEVICE)
    return model


def get_data():
    _image_size = 224
    _mean, _std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
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

    train_ds = DogsCatsDataset("../data/raw", "sample/train", transform=train_trans)
    val_ds = DogsCatsDataset("../data/raw", "sample/valid", transform=val_trans)

    batch_size = 2
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    return train_dl, val_dl
