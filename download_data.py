"""
Download all the data that is needed for the tutorial
so we don't crash the conference network.

"""
from pathlib import Path

from torchvision import models
from torchvision.datasets import MNIST, CIFAR10

from notebooks import my_datasets
from notebooks.utils import ptitle


if __name__ == "__main__":
    ROOT = Path("data/raw")
    ROOT.mkdir(parents=True, exist_ok=True)

    ptitle("Downloading DogsCatsDataset")
    _ds = my_datasets.DogsCatsDataset(ROOT, "train", download=True)

    print()
    ptitle("Downloading MNIST")
    _ds = MNIST(ROOT, train=True, download=True)
    _ds = MNIST(ROOT, train=False, download=True)

    print()
    ptitle("Downloading CIFAR10")
    _ds = CIFAR10(ROOT, train=True, download=True)
    _ds = CIFAR10(ROOT, train=False, download=True)

    print()
    ptitle("Downloading models")
    _model = models.resnet18(pretrained=True)
    _model = models.squeezenet1_1(pretrained=True)
