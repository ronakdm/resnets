import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

import copy
import time

def get_param_shapes(parameters):
    return [torch.tensor(p.shape) for p in parameters]

def get_num_parameters(param_shapes):
    return torch.tensor([torch.prod(s) for s in param_shapes]).sum().item()

def unflatten_gradient(g, param_shapes):
    chunks = torch.split(g, [torch.prod(s) for s in param_shapes])
    for i, chunk in enumerate(chunks):
        chunk = chunk.reshape(tuple(param_shapes[i]))
    return chunks

def flatten_parameters(parameters):
    return torch.cat([p.reshape(-1) for p in parameters])

class ConvNet(nn.Module):

    def __init__(self, height, width, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=1, padding='same')
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding='same')
        self.bn2 = nn.BatchNorm2d(128)
        # self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding='same')
        # self.bn3 = nn.BatchNorm2d(256)
        # self.classifier = nn.Linear(height * width * 256, n_classes)
        self.classifier = nn.Linear(height * width * 128, n_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        return self.classifier(torch.flatten(x, start_dim=1))
    
class GradientModel(nn.Module):

    def __init__(self, height, width, n_classes, n_parameters, hidden_size=512):
        super().__init__()

        # image backbone
        channels = 32
        # img_embed_dim = height * width * 2 * channels
        img_embed_dim = height * width * channels
        self.image_embed = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=(3, 3), stride=1, padding='same'),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            # nn.Conv2d(channels, 2 * channels, kernel_size=(3, 3), stride=1, padding='same'),
            # nn.BatchNorm2d(2 * channels),
            # nn.ReLU(),
            nn.Flatten(start_dim=1)
        )

        # label embedding
        label_embed_dim = 64
        self.label_embed = nn.Embedding(n_classes, label_embed_dim)

        # feed forward
        self.head = nn.Sequential(
            nn.Linear(img_embed_dim + label_embed_dim + n_parameters, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_parameters),
        )

    def forward(self, x, y, w):
        n = len(x)
        img = self.image_embed(x)
        lab = self.label_embed(y)
        w = w.reshape(1, -1).expand([n, len(w)]) # copy to be n x d without additional memory
        h = self.head(torch.cat([img, lab, w], dim=1))
        return torch.mean(h, axis=0)
    

def train_one_epoch_func(dataloader, func_model, func_optimizer):
    floss = 0.0
    for i, (x, y) in enumerate(dataloader):
        logits = func_model(x.to(DEVICE))
        loss = F.cross_entropy(logits, y.to(DEVICE))
        loss.backward()
        func_optimizer.step()
        func_optimizer.zero_grad()
        floss += loss.item() / len(dataloader)
    return floss

@torch.no_grad()
def evaluate_func(dataloader, func_model):
    floss = 0.0
    acc = 0.0
    for i, (x, y) in enumerate(dataloader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = func_model(x)
        loss = F.cross_entropy(logits, y)
        floss += loss.item() / len(dataloader)
        acc += torch.mean((y == torch.argmax(logits, dim=1)).float()).item() / len(dataloader)
    return floss, acc

def train_one_epoch_grad(dataloader, func_model, grad_model, grad_optimizer):
    gloss = 0.0
    param = flatten_parameters(func_model.parameters()).clone()
    for i, (x, y) in enumerate(dataloader):
        logits = func_model(x.to(DEVICE))
        loss = F.cross_entropy(logits, y.to(DEVICE))
        grads = torch.autograd.grad(outputs=loss, inputs=func_model.parameters())

        targets_true = flatten_parameters(grads).to(DEVICE)
        targets_pred = grad_model(x.to(DEVICE), y.to(DEVICE), param)
        error = F.mse_loss(targets_pred, targets_true)
        error.backward()

        grad_optimizer.step()
        grad_optimizer.zero_grad()
        gloss += error.item() / len(dataloader)
    return gloss

def evaluate_grad(dataloader, func_model, grad_model):
    gloss = 0.0
    param = flatten_parameters(func_model.parameters()).clone()
    for i, (x, y) in enumerate(dataloader):
        logits = func_model(x.to(DEVICE))
        loss = F.cross_entropy(logits, y.to(DEVICE))
        grads = torch.autograd.grad(outputs=loss, inputs=func_model.parameters())
        targets_true = flatten_parameters(grads).to(DEVICE)
        with torch.no_grad():
            targets_pred = grad_model(x.to(DEVICE), y.to(DEVICE), param)
            error = F.mse_loss(targets_pred, targets_true)
        gloss += error.item() / len(dataloader)
    return gloss


if __name__ == "__main__":
    DEVICE = "cuda:1"
    DATA_PATH = "/mnt/ssd/ronak/datasets/"

    root = os.path.join(DATA_PATH, "fashion_mnist")
    train_set = FashionMNIST(root, download=True)
    mean = train_set.data.float().mean(dim=0)
    std = train_set.data.float().std(dim=0)

    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ])

    # reload datasets
    train_set = FashionMNIST(root, download=True, transform=transform)
    val_set = FashionMNIST(root, download=True, transform=transform, train=False)

    n_epochs = 10
    batch_size = 128
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    height, width, n_classes = 28, 28, 10

    func_model = ConvNet(height, width, n_classes).to(DEVICE)
    param_shapes = get_param_shapes(func_model.parameters())
    n_parameters = get_num_parameters(param_shapes)
    grad_model = GradientModel(height, width, n_classes, n_parameters, hidden_size=64).to(DEVICE)

    func_optimizer = torch.optim.Adam(func_model.parameters(), lr=1e-3)
    grad_optimizer = torch.optim.Adam(grad_model.parameters(), lr=1e-4)

    func_train_loss = torch.zeros(n_epochs)
    grad_train_loss = torch.zeros(n_epochs)
    func_val_acc    = torch.zeros(n_epochs)
    grad_val_loss   = torch.zeros(n_epochs)

    torch.manual_seed(123)
    for epoch in range(n_epochs):
        # train the function model for one epoch to learn the parameters
        func_train_loss[epoch] = train_one_epoch_func(train_loader, func_model, func_optimizer)
        print(f"func train loss epoch {epoch:02d}: {func_train_loss[epoch]:0.5f}")
        _, func_val_acc[epoch] = evaluate_func(val_loader, func_model)
        print(f"func valid acc epoch {epoch:02d}:  {func_val_acc[epoch]:0.5f}")

        # train the gradient model for one epoch
        grad_train_loss[epoch] = train_one_epoch_grad(train_loader, func_model, grad_model, grad_optimizer)
        print(f"grad train loss epoch {epoch:02d}: {grad_train_loss[epoch]:0.5f}")
        grad_val_loss[epoch] = evaluate_grad(val_loader, func_model, grad_model)
        print(f"grad valid loss epoch {epoch:02d}: {grad_val_loss[epoch]:0.5f}")
        print()