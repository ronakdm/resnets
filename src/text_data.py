import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
import numpy as np
import os

class TokenizedTextClassificationDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x).long()
        self.y = torch.tensor(y).float()
        self.n = len(self.x)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i, self.x[i], self.y[i]

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
    
def get_text_dataloaders(
    batch_size, rank, n_bins=50, factor=1.0, root="/mnt/ssd/ronak/datasets/sst2"
):
    x_train = np.load(os.path.join(root, "x_train.npy"))
    y_train = np.load(os.path.join(root, "y_train.npy"))
    x_test  = np.load(os.path.join(root, "x_val.npy"))
    y_test  = np.load(os.path.join(root, "y_val.npy"))

    model_name = "vit_b32_laion2b"
    quantization = {
        "x_labels":   np.load(os.path.join(root, f"quantization/{model_name}_kmeans_{n_bins}/text_labels.npy")),
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


    train_dataset = TokenizedTextClassificationDataset(x_train, y_train)
    train_dataloader = DataLoader(
        train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size
    )
    print(f"{len(train_dataset):>5,} training samples on rank {rank}.")
    test_dataset = TokenizedTextClassificationDataset(x_test, y_test)
    test_dataloader = DataLoader(
        test_dataset, sampler=RandomSampler(test_dataset), batch_size=batch_size
    )
    print(f"{len(test_dataset):>5,} validation samples on rank {rank}.")
    return train_dataloader, test_dataloader, quantization