"""
Download all the data that is needed for the tutorial
so we don't crash the conference network.

"""
from torchvision import models
from torchvision.datasets import MNIST, CIFAR10
from ppt import utils


if __name__ == "__main__":
    ROOT = "data/raw"

    print("=" * 80)
    print("# Downloading DogsCatsDataset")
    ds = utils.DogsCatsDataset(ROOT, "train", download=True)

    print()
    print("=" * 80)
    print("# Downloading MNIST")
    ds = MNIST(ROOT, train=True, download=True)
    ds = MNIST(ROOT, train=False, download=True)

    print()
    print("=" * 80)
    print("# Downloading MNIST")
    ds = CIFAR10(ROOT, train=True, download=True)
    ds = CIFAR10(ROOT, train=False, download=True)

    print()
    print("=" * 80)
    print("# Downloading models")
    model = models.resnet18(pretrained=True)
    model = models.squeezenet1_1(pretrained=True)
