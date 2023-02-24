import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader, RandomSampler

class ImageClassificationDataset(Dataset):

    def __init__(self, x, y):
        self.x = torch.tensor(x).float()
        self.y = torch.tensor(y).long()
        self.sample_size = len(self.x)

    def __len__(self):
        return self.sample_size

    def __getitem__(self, i):
        return self.x[i], self.y[i]

def preprocess(x_tr, x_te):

    # Pad
    # border = 4
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

    # Transpose to put channels before height and width.
    x_tr = x_tr.transpose(0, 3, 1, 2)
    x_te = x_te.transpose(0, 3, 1, 2)

    # Randomly flip left and write.
    np.random.seed(123)
    n = len(x_tr)
    flips = np.random.binomial(1, 0.5, size=(n))
    x_tr = np.array([x_tr[i][..., ::-1] if flips[i] else x_tr[i] for i in range(n)])

    return x_tr, x_te


def load_cifar10(root="data/"):

    # Get train data.
    train_data = CIFAR10(root, download=True)
    test_data = CIFAR10(root, train=False, download=True)

    x_train = train_data.data
    x_test = test_data.data
    y_train = np.array(train_data.targets)
    y_test = np.array(test_data.targets)

    # Apply preprocessing.
    x_train, x_test = preprocess(x_train, x_test)

    return x_train, y_train, x_test, y_test

def get_cifar10_loaders(batch_size, root="data/"):
    x_train, y_train, x_test, y_test = load_cifar10()
    train_dataset = ImageClassificationDataset(x_train, y_train)
    train_dataloader = DataLoader(
        train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size
    )
    print("{:>5,} training samples.".format(len(train_dataset)))
    test_dataset = ImageClassificationDataset(x_test, y_test)
    test_dataloader = DataLoader(
        test_dataset, sampler=RandomSampler(test_dataset), batch_size=batch_size
    )
    print("{:>5,} test samples.".format(len(test_dataset)))
    return train_dataloader, test_dataloader