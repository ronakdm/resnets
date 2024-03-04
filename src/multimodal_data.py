import torch
import os
from torch.utils.data import Dataset, DataLoader

class MultimodalEmbeddingDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.n = len(self.x)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i, self.x[i], self.y[i]

def get_multimodal_dataloaders(
    batch_size, rank, unbalance=1.0, root="/mnt/ssd/ronak/datasets/imagenet_captions_50k", quantization=None,
):
    x_train = torch.load(os.path.join(root, "x_train.pt"))
    y_train = torch.load(os.path.join(root, "y_train.pt"))
    x_test  = torch.load(os.path.join(root, "x_test.pt"))
    y_test  = torch.load(os.path.join(root, "y_test.pt"))

    train_dataset = MultimodalEmbeddingDataset(x_train, y_train)
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size
    )
    print(f"{len(train_dataset):>5,} training samples on rank {rank}.")
    test_dataset = MultimodalEmbeddingDataset(x_test, y_test)
    test_dataloader = DataLoader(
        test_dataset, shuffle=True, batch_size=batch_size
    )
    print(f"{len(test_dataset):>5,} validation samples on rank {rank}.")
    return train_dataloader, test_dataloader, quantization