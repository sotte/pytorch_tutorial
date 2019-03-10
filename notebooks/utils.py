import os
import zipfile

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models
from torchvision.datasets.folder import ImageFolder, default_loader
from torchvision.datasets.utils import download_url, check_integrity

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
# PyTorch
class DogsCatsDataset(ImageFolder):
    """
    The 'Dogs and Cats' dataset from kaggle.

    https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/

    Args:
        root: the location where to store the dataset
        suffix: path to the train/valid/sample dataset. See folder structure.
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that
            takes in the target and transforms it.
        loader: A function to load an image given its path.
        download: if ``True``, download the data.


    The folder structure of the dataset is as follows::

        └── dogscats
            ├── sample
            │   ├── train
            │   │   ├── cats
            │   │   └── dogs
            │   └── valid
            │       ├── cats
            │       └── dogs
            ├── train
            │   ├── cats
            │   └── dogs
            └── valid
                ├── cats
                └── dogs

    """

    url = "http://files.fast.ai/data/dogscats.zip"
    filename = "dogscats.zip"
    checksum = "aef22ec7d472dd60e8ee79eecc19f131"

    def __init__(
        self,
        root: str,
        suffix: str,
        transform=None,
        target_transform=None,
        loader=default_loader,
        download=False,
    ):
        self.root = os.path.expanduser(root)

        if download:
            self._download()
            self._extract()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. "
                "You can use download=True to download it"
            )

        path = os.path.join(self.root, "dogscats", suffix)
        print(f"Loading data from {path}.")
        assert os.path.isdir(path), f"'{suffix}' is not valid."

        super().__init__(path, transform, target_transform, loader)

    def _download(self):
        if self._check_integrity():
            print("Dataset already downloaded and verified.")
            return

        root = self.root
        print("Downloading dataset... (this might take a while)")
        download_url(self.url, root, self.filename, self.checksum)

    def _extract(self):
        path_to_zip = os.path.join(self.root, self.filename)
        with zipfile.ZipFile(path_to_zip, "r") as zip_ref:
            zip_ref.extractall(self.root)

    def _check_integrity(self):
        path_to_zip = os.path.join(self.root, self.filename)
        return check_integrity(path_to_zip, self.checksum)


# Training helpers
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


################################################################################
# DOGS AND CATS DEMO
def get_model(n_classes=2):
    model = models.resnet18(pretrained=True)
    freeze_all(model.parameters())
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
