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

class ImageClassificationDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = torch.tensor(x).float()
        self.y = torch.tensor(y).long()
        self.n = len(self.x)
        self.transform = transform

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.transform is None:
            return i, self.x[i], self.y[i]
        else:
            return i, self.transform(self.x[i]), self.y[i]

def preprocess(x_tr, x_te, border=4):
    # Pad
    # x_tr = np.pad(
    #     x_tr, [(0, 0), (border, border), (border, border), (0, 0)], mode="reflect"
    # )
    # x_te = np.pad(
    #     x_te, [(0, 0), (border, border), (border, border), (0, 0)], mode="reflect"
    # )

    # Normalize.
    mean, std = x_tr.mean(axis=0), x_tr.std(axis=0)
    x_tr = (x_tr - mean) / std
    x_te = (x_te - mean) / std

    return x_tr, x_te

def rebalance(y, factor):
    classes, class_sizes = np.unique(y, return_counts=True)
    n_classes = len(classes)
    weights = np.linspace(1, factor, n_classes)
    weights /= weights.sum()
    down_sizes = np.round(weights * class_sizes[-1] / weights.max()).astype(int)
    all_idx = np.arange(len(y))
    new_idx = []
    np.random.seed(123)
    for c in classes:
        new_idx.append(np.random.choice(all_idx[y == c], down_sizes[c], replace=False))
    return np.concatenate(new_idx, axis=0)


def get_image_dataloaders(
    batch_size, rank, n_bins=50, factor=1.0, root="/mnt/ssd/ronak/datasets/cifar10"
):
    x_train = np.load(os.path.join(root, "x_train.npy"))
    y_train = np.load(os.path.join(root, "y_train.npy"))
    x_test  = np.load(os.path.join(root, "x_test.npy"))
    y_test  = np.load(os.path.join(root, "y_test.npy"))

    model_name = 'convnext_base'
    quantization = {
        "x_labels":   np.load(os.path.join(root, f"quantization/{model_name}_kmeans_{n_bins}/image_labels.npy")),
        "y_labels":   np.load(os.path.join(root, f"quantization/{model_name}_kmeans_{n_bins}/class_labels.npy")),
    }

    # reindex for class imbalance
    if factor > 1.0:
        idx = rebalance(y_train, factor)
        print(f"rebalancing to factor {factor}...")
        x_train = x_train[idx]
        y_train = y_train[idx]
        print(f"train features shape {x_train.shape}")
        print(f"train labels shape {x_train.shape}")
        quantization["x_labels"] = quantization["x_labels"][idx]
        quantization["y_labels"] = quantization["y_labels"][idx]

    quantization["x_marginal"] = np.unique(quantization["x_labels"], return_counts=True)[1] / len(x_train)
    quantization["y_marginal"] = np.unique(quantization["y_labels"], return_counts=True)[1] / len(y_train)

    # Apply preprocessing.
    x_train, x_test = preprocess(x_train, x_test)

    # Transpose to put channels before height and width.
    x_train = x_train.transpose(0, 3, 1, 2)
    x_test = x_test.transpose(0, 3, 1, 2)

    # augmentation
    # crop = x_train.shape[-1]
    # cutout = crop // 4
    # train_transform = Compose(
    #     [RandomCrop(crop), RandomHorizontalFlip(0.5), Cutout(cutout)]
    # )
    # no augmentation
    # test_transform = CenterCrop(crop)

    train_dataset = ImageClassificationDataset(x_train, y_train)
    train_dataloader = DataLoader(
        train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size
    )
    print(f"{len(train_dataset):>5,} training samples on rank {rank}.")
    test_dataset = ImageClassificationDataset(x_test, y_test)
    test_dataloader = DataLoader(
        test_dataset, sampler=RandomSampler(test_dataset), batch_size=batch_size
    )
    print(f"{len(test_dataset):>5,} validation samples on rank {rank}.")
    return train_dataloader, test_dataloader, quantization
