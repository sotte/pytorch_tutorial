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
