"""Download common deep learning datasets to data/."""

from pathlib import Path

import torchvision.datasets as datasets

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data"

DATA_PATH.mkdir(exist_ok=True)

cifar10_train = datasets.CIFAR10(DATA_PATH / "cifar10", train=True, download=True)
cifar10_test = datasets.CIFAR10(DATA_PATH / "cifar10", train=False, download=True)

cifar100_train = datasets.CIFAR100(DATA_PATH / "cifar100", train=True, download=True)
cifar100_test = datasets.CIFAR100(DATA_PATH / "cifar100", train=False, download=True)

mnist_train = datasets.MNIST(DATA_PATH / "mnist", train=True, download=True)
mnist_test = datasets.MNIST(DATA_PATH / "mnist", train=False, download=True)

fashion_train = datasets.FashionMNIST(DATA_PATH / "fashion_mnist", train=True, download=True)
fashion_test = datasets.FashionMNIST(DATA_PATH / "fashion_mnist", train=False, download=True)

tiny_imagenet_path = DATA_PATH / "tiny-imagenet-200"
tiny_imagenet_train = datasets.ImageFolder(tiny_imagenet_path / "train")
tiny_imagenet_val = datasets.ImageFolder(tiny_imagenet_path / "val")
tiny_imagenet_test = datasets.ImageFolder(tiny_imagenet_path / "test")

print("Datasets downloaded to data/!")
print("CIFAR-10:", len(cifar10_train), "train samples,", len(cifar10_test), "test samples")
print("CIFAR-100:", len(cifar100_train), "train samples,", len(cifar100_test), "test samples")
print("MNIST:", len(mnist_train), "train samples,", len(mnist_test), "test samples")
print("Fashion-MNIST:", len(fashion_train), "train samples,", len(fashion_test), "test samples")
print("Tiny ImageNet:", len(tiny_imagenet_train), "train samples,", len(tiny_imagenet_val), "validation samples,", len(tiny_imagenet_test), "test samples")
