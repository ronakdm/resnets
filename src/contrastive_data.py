import torch
import os
from torch.utils.data import TensorDataset, DataLoader

def get_contrastive_dataloaders(
    batch_size, rank, unbalance=1.0, root="/mnt/ssd/ronak/datasets/imagenet_captions_50k", quantization=None,
):
    x_train = torch.load(os.path.join(root, "x_train.npy"))
    y_train = torch.load(os.path.join(root, "y_train.npy"))
    x_test  = torch.load(os.path.join(root, "x_test.npy"))
    y_test  = torch.load(os.path.join(root, "y_test.npy"))

    train_dataset = TensorDataset(x_train, y_train)
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size
    )
    print(f"{len(train_dataset):>5,} training samples on rank {rank}.")
    test_dataset = TensorDataset(x_test, y_test)
    test_dataloader = DataLoader(
        test_dataset, shuffle=True, batch_size=batch_size
    )
    print(f"{len(test_dataset):>5,} validation samples on rank {rank}.")
    return train_dataloader, test_dataloader, quantization