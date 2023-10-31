import numpy as np
import torch
import os

# from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, CenterCrop
from torch.utils.data import Dataset, DataLoader, RandomSampler


class Cutout:
    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        x0, y0 = torch.randint(x.shape[-2] - self.size, (1,)), torch.randint(
            x.shape[-1] - self.size, (1,)
        )
        img = x.clone()
        img[..., y0 : y0 + self.size, x0 : x0 + self.size] = 0.0
        return img


# class ImageClassificationDataLoader:
#     def __init__(
#         self,
#         features,
#         labels,
#         center=None,
#         crop=32,
#         flip=0.5,
#         cutout=8,
#     ):
#         if center:
#             self.pipeline = CenterCrop(center)
#         else:
#             pipeline = []
#             for param, transform in zip(
#                 [crop, flip, cutout], [RandomCrop, RandomHorizontalFlip, Cutout]
#             ):
#                 if param:
#                     pipeline.append(transform(param))
#             self.pipeline = Compose(pipeline)

#         self.features = torch.from_numpy(features).float()
#         self.labels = torch.from_numpy(labels).long()

#     def get_batch(self, batch_size, device):
#         ix = torch.randint(len(self.features), (batch_size,))
#         # ix = torch.arange(batch_size)
#         x = self.pipeline(self.features[ix])
#         y = self.labels[ix]
#         if device != "cpu":
#             # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
#             x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
#                 device, non_blocking=True
#             )
#         return x, y


class ImageClassificationDataset(Dataset):
    def __init__(self, x, y, augment=False):
        self.x = torch.tensor(x).float()
        self.y = torch.tensor(y).long()
        self.n = len(self.x)
        if augment:
            self.pipeline = Compose(
                [RandomCrop(32), RandomHorizontalFlip(0.5), Cutout(8)]
            )
        else:
            self.pipeline = CenterCrop(32)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.pipeline(self.x[i]), self.y[i]


class QuantizedImageClassificationDataset(Dataset):
    def __init__(self, x, y, cx, cy, augment=False):
        self.x = torch.tensor(x).float()
        self.y = torch.tensor(y).long()
        self.cx = torch.tensor(cx).long()
        self.cy = torch.tensor(cy).long()
        self.n = len(self.x)
        if augment:
            self.pipeline = Compose(
                [RandomCrop(32), RandomHorizontalFlip(0.5), Cutout(8)]
            )
        else:
            self.pipeline = CenterCrop(32)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.pipeline(self.x[i]), self.y[i], self.cx[i], self.cy[i]


def preprocess(x_tr, x_te, border=4):
    # Pad
    x_tr = np.pad(
        x_tr, [(0, 0), (border, border), (border, border), (0, 0)], mode="reflect"
    )
    x_te = np.pad(
        x_te, [(0, 0), (border, border), (border, border), (0, 0)], mode="reflect"
    )

    # Normalize.
    mean, std = x_tr.mean(axis=0), x_tr.std(axis=0)
    x_tr = (x_tr - mean) / std
    x_te = (x_te - mean) / std

    # Transpose to put channels before height and width.
    x_tr = x_tr.transpose(0, 3, 1, 2)
    x_te = x_te.transpose(0, 3, 1, 2)

    return x_tr, x_te


# def compute_image_classification_metrics(model, X, Y):
#     loss, logits = model(X, Y)
#     accuracy = torch.sum((torch.argmax(logits, dim=1) == Y)) / len(Y)
#     return {"loss": loss, "accuracy": accuracy}


# def load_cifar10(root="/mnt/ssd/ronak/datasets/"):
#     # Get train data.
#     train_data = CIFAR10(root, download=True)
#     test_data = CIFAR10(root, train=False, download=True)

#     x_train = train_data.data
#     x_test = test_data.data
#     y_train = np.array(train_data.targets)
#     y_test = np.array(test_data.targets)

#     # Apply preprocessing.
#     x_train, x_test = preprocess(x_train, x_test)

#     train_loader = ImageClassificationDataLoader(x_train, y_train)
#     test_loader = ImageClassificationDataLoader(x_test, y_test, center=32)

#     return train_loader, test_loader, compute_image_classification_metrics


def get_cifar10_loaders(
    batch_size, rank, n_bins, root="/mnt/ssd/ronak/datasets/cifar10"
):
    # Get train data.
    # train_data = CIFAR10(root, download=True)
    # test_data = CIFAR10(root, train=False, download=True)

    # x_train = train_data.data
    # x_test = test_data.data
    # y_train = np.array(train_data.targets)
    # y_test = np.array(test_data.targets)

    x_train = np.load(os.path.join(root, "x_train.npy"))
    y_train = np.load(os.path.join(root, "y_train.npy"))
    x_test = np.load(os.path.join(root, "x_test.npy"))
    y_test = np.load(os.path.join(root, "y_test.npy"))

    # Apply preprocessing.
    x_train, x_test = preprocess(x_train, x_test)

    train_dataset = ImageClassificationDataset(x_train, y_train, augment=True)
    train_dataloader = DataLoader(
        train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size
    )
    print(f"{len(train_dataset):>5,} training samples on rank {rank}.")
    test_dataset = ImageClassificationDataset(x_test, y_test, augment=False)
    test_dataloader = DataLoader(
        test_dataset, sampler=RandomSampler(test_dataset), batch_size=batch_size
    )
    print(f"{len(test_dataset):>5,} validation samples on rank {rank}.")
    return train_dataloader, test_dataloader


def get_quantized_cifar10_loaders(
    batch_size, rank, n_bins=40, root="/mnt/ssd/ronak/datasets/cifar10"
):
    x_train = np.load(os.path.join(root, "x_train.npy"))
    y_train = np.load(os.path.join(root, "y_train.npy"))
    x_test = np.load(os.path.join(root, "x_test.npy"))
    y_test = np.load(os.path.join(root, "y_test.npy"))

    cx_train = np.load(os.path.join(root, f"x_train_quantized_{n_bins}.npy"))
    cy_train = np.load(os.path.join(root, f"y_train_quantized_{n_bins}.npy"))
    cx_test = np.load(os.path.join(root, f"x_test_quantized_{n_bins}.npy"))
    cy_test = np.load(os.path.join(root, f"y_test_quantized_{n_bins}.npy"))

    px = np.load(os.path.join(root, "x_marginal.npy"))
    py = np.load(os.path.join(root, "y_marginal.npy"))
    marginals = (px, py)

    # Apply preprocessing.
    x_train, x_test = preprocess(x_train, x_test)

    train_dataset = QuantizedImageClassificationDataset(
        x_train, y_train, cx_train, cy_train, augment=True
    )
    train_dataloader = DataLoader(
        train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size
    )
    print(f"{len(train_dataset):>5,} training samples on rank {rank}.")
    test_dataset = QuantizedImageClassificationDataset(
        x_test, y_test, cx_test, cy_test, augment=False
    )
    test_dataloader = DataLoader(
        test_dataset, sampler=RandomSampler(test_dataset), batch_size=batch_size
    )
    print(f"{len(test_dataset):>5,} validation samples on rank {rank}.")
    return train_dataloader, test_dataloader, marginals
