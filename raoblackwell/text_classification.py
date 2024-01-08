import os
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

import copy
import time

import sys
sys.path.extend(["..", "."])
from src.text_data import TokenizedTextClassificationDataset

DEVICE = "cuda:1"
DATA_PATH = "/mnt/ssd/ronak/datasets/"
SCORE_MODEL = "gpt2"


def get_param_shapes(parameters):
    return [torch.tensor(p.shape) for p in parameters]

def get_num_parameters(param_shapes):
    return torch.tensor([torch.prod(s) for s in param_shapes]).sum().item()

def unflatten_gradient(g, param_shapes):
    chunks = list(torch.split(g, [torch.prod(s) for s in param_shapes]))
    for i, chunk in enumerate(chunks):
        chunks[i] = chunk.reshape(tuple(param_shapes[i]))
    return chunks

def flatten_parameters(parameters):
    return torch.cat([p.reshape(-1) for p in parameters])

# vocab size is the CLIP tokenizer by default
class RecurrentNet(nn.Module):

    def __init__(self, n_classes, vocab_size=49408):
        super().__init__()
        embed_dim = 32
        hidden_dim = 32
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm1 = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True)
        # self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        if n_classes == 2:
            self.classifier = nn.Linear(hidden_dim, 1)
        else:
            self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        h = self.embed(x)
        h = self.lstm1(h)
        # h = self.lstm2(h)
        out = self.classifier(h[0][:, -1, :]) # take last hidden state in the sequence
        return out.squeeze()
    
class GradientModel(nn.Module):

    def __init__(self, n_model_params, n_score_params, hidden_size=512):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(n_model_params + n_score_params, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_model_params),
        )

    def forward(self, g, score):
        # reshape gradient to match the number of scores (batch size)
        g_ = g.reshape(1, -1).expand([len(score), len(g)])
        x = torch.cat([g_, score], dim=1)
        return torch.mean(self.ffn(x), axis=0) + g
    

def train_one_epoch_func(dataloader, func_model, func_optimizer):
    
    floss = 0.0
    acc = 0.0
    for i, (idx, x, y) in enumerate(dataloader):
        logits = func_model(x.to(DEVICE))
        loss = F.binary_cross_entropy_with_logits(logits, y.to(DEVICE))
        loss.backward()
        func_optimizer.step()
        func_optimizer.zero_grad()
        floss += loss.item() / len(dataloader)
        acc += torch.mean((y.squeeze().to(DEVICE) == (logits.squeeze() >= 0).int()).float()).item() / len(dataloader)
    return floss, acc

@torch.no_grad()
def evaluate_func(dataloader, func_model):
    
    floss = 0.0
    acc = 0.0
    for i, (idx, x, y) in enumerate(dataloader):
        logits = func_model(x.to(DEVICE))
        loss = F.binary_cross_entropy_with_logits(logits, y.to(DEVICE))
        floss += loss.item() / len(dataloader)
        acc += torch.mean((y.squeeze().to(DEVICE) == (logits.squeeze() >= 0).int()).float()).item() / len(dataloader)
    return floss, acc

def train_one_epoch_grad(dataloader, func_model, grad_model, grad_optimizer, scores):
    
    
    param_shapes = get_param_shapes(func_model.parameters())

    # Use one epoch to compute the full batch gradients.
    full_grad = [torch.zeros(tuple(s), requires_grad=False).to(DEVICE) for s in param_shapes]
    for i, (idx, x, y) in enumerate(dataloader):
        func_model.zero_grad()
        logits = func_model(x.to(DEVICE))
        loss = F.binary_cross_entropy_with_logits(logits, y.to(DEVICE))
        grads = torch.autograd.grad(outputs=loss, inputs=func_model.parameters())
        with torch.no_grad():
            for sgrad, fgrad in zip(grads, full_grad):
                fgrad += sgrad.detach() / len(dataloader)
    
    # Use a second epoch to train the gradient model.
    gloss = 0.0
    for i, (idx, x, y) in enumerate(dataloader):

        logits = func_model(x.to(DEVICE))
        loss = F.binary_cross_entropy_with_logits(logits, y.to(DEVICE))
        grads = torch.autograd.grad(outputs=loss, inputs=func_model.parameters())

        targets_true = flatten_parameters(full_grad)
        targets_pred = grad_model(flatten_parameters(grads), scores[idx])
        error = F.mse_loss(targets_pred, targets_true)
        error.backward()

        grad_optimizer.step()
        grad_optimizer.zero_grad()
        gloss += error.item() / len(dataloader)
    return gloss

@torch.no_grad()
def evaluate_grad(dataloader, func_model, grad_model, scores):
    
    
    param_shapes = get_param_shapes(func_model.parameters())
    full_grad = [torch.zeros(tuple(s)).to(DEVICE) for s in param_shapes]
    # Use one epoch to compute the full batch gradients.
    for i, (idx, x, y) in enumerate(dataloader):
        func_model.zero_grad()
        with torch.enable_grad():
            logits = func_model(x.to(DEVICE))
            loss = F.binary_cross_entropy_with_logits(logits, y.to(DEVICE))
            grads = torch.autograd.grad(outputs=loss, inputs=func_model.parameters())
        for sgrad, fgrad in zip(grads, full_grad):
            fgrad += sgrad.detach() / len(dataloader)

    gloss = 0.0
    for i, (idx, x, y) in enumerate(dataloader):
        with torch.enable_grad():
            logits = func_model(x.to(DEVICE))
            loss = F.binary_cross_entropy_with_logits(logits, y.to(DEVICE))
            grads = torch.autograd.grad(outputs=loss, inputs=func_model.parameters())

        targets_true = flatten_parameters(full_grad)
        targets_pred = grad_model(flatten_parameters(grads), scores[idx])
        error = F.mse_loss(targets_pred, targets_true)
        gloss += error.item() / len(dataloader)
    return gloss


# TRAINING

root = os.path.join(DATA_PATH, "sst2")
# flip so that padding is on the left for LSTM
x_train = np.flip(np.load(os.path.join(root, "x_train.npy")), axis=1).copy()
y_train = np.load(os.path.join(root, "y_train.npy"))
x_test  = np.flip(np.load(os.path.join(root, "x_val.npy")), axis=1).copy()
y_test  = np.load(os.path.join(root, "y_val.npy"))

for dat, name in zip([x_train, y_train, x_test, y_test], ["x_train", "y_train", "x_test", "y_test"]):
    print(f"{name} shape: {dat.shape}")

# load original dataset
batch_size = 128
train_dataset = TokenizedTextClassificationDataset(x_train, y_train)
train_loader = DataLoader(
    train_dataset, shuffle=True, batch_size=batch_size
)
print(f"{len(train_dataset):>5,} training samples.")
val_dataset = TokenizedTextClassificationDataset(x_test, y_test)
val_loader = DataLoader(
    val_dataset, shuffle=True, batch_size=batch_size
)
print(f"{len(val_dataset):>5,} validation samples.")

# load scores

scores = torch.from_numpy(np.load(os.path.join(root, f"{SCORE_MODEL}_scores.npy"))).to(DEVICE)
print(f"scores shape: {tuple(scores.shape)}")

# load models

n_epochs = 10
n_classes = 2

func_model = RecurrentNet(n_classes).to(DEVICE)
param_shapes = get_param_shapes(func_model.parameters())
# n_parameters = get_num_parameters(param_shapes)
n_model_params, n_score_params = get_num_parameters(param_shapes), scores.shape[1]
print(f"number of parameters {n_model_params}")
grad_model = GradientModel(n_model_params, n_score_params, hidden_size=16).to(DEVICE)

func_optimizer = torch.optim.SGD(func_model.parameters(), lr=1.0)
grad_optimizer = torch.optim.AdamW(grad_model.parameters(), lr=1e-4, weight_decay=30.0)

# train

func_train_loss = torch.zeros(n_epochs)
func_train_acc  = torch.zeros(n_epochs)
func_val_loss   = torch.zeros(n_epochs)
func_val_acc    = torch.zeros(n_epochs)

grad_train_loss = torch.zeros(n_epochs)
grad_val_loss   = torch.zeros(n_epochs)



torch.manual_seed(123)
for epoch in range(n_epochs):
    # train the function model for one epoch to learn the parameters
    func_train_loss[epoch], func_train_acc[epoch] = train_one_epoch_func(train_loader, func_model, func_optimizer)
    print(f"func train loss epoch {epoch:02d}: {func_train_loss[epoch]:0.5f}")
    func_val_loss[epoch], func_val_acc[epoch] = evaluate_func(val_loader, func_model)
    print(f"func valid acc epoch {epoch:02d}:  {func_val_acc[epoch]:0.5f}")

    # train the gradient model for one epoch
    grad_train_loss[epoch] = train_one_epoch_grad(train_loader, func_model, grad_model, grad_optimizer, scores)
    print(f"grad train loss epoch {epoch:02d}: {grad_train_loss[epoch]:0.9f}")
    grad_val_loss[epoch] = evaluate_grad(val_loader, func_model, grad_model, scores)
    print(f"grad valid loss epoch {epoch:02d}: {grad_val_loss[epoch]:0.9f}")
    print()

torch.save(grad_model.state_dict(), os.path.join(root, "grad_model.pt"))
torch.save(func_train_loss, "raoblackwell/sst2_sgd_func_train_loss.pt")
torch.save(func_train_acc, "raoblackwell/sst2_sgd_func_train_acc.pt")
torch.save(func_val_loss, "raoblackwell/sst2_sgd_func_val_loss.pt")
torch.save(func_val_acc, "raoblackwell/sst2_sgd_func_val_acc.pt")
torch.save(grad_train_loss, "raoblackwell/sst2_sgd_grad_train_loss.pt")
torch.save(grad_val_loss, "raoblackwell/sst2_sgd_grad_val_loss.pt")

# EVALUATION

DEVICE = "cuda:2"
n_classes = 2
root = os.path.join(DATA_PATH, "sst2")

scores = torch.from_numpy(np.load(os.path.join(root, f"{SCORE_MODEL}_scores.npy"))).to(DEVICE)
print(f"scores shape: {tuple(scores.shape)}")

model = RecurrentNet(n_classes).to(DEVICE)
param_shapes = get_param_shapes(model.parameters())
n_model_params, n_score_params = get_num_parameters(param_shapes), scores.shape[1]
print(f"number of parameters {n_model_params}")

gmodel = GradientModel(n_model_params, n_score_params, hidden_size=16).to(DEVICE)
gmodel.load_state_dict(torch.load(os.path.join(root, "grad_model.pt")))

# foptimizer = torch.optim.SGD(model.parameters(), lr=0.1)
goptimizer = torch.optim.Adam(gmodel.parameters(), lr=1e-4)

# flip so that padding is on the left for LSTM
x_train = np.flip(np.load(os.path.join(root, "x_train.npy")), axis=1).copy()
y_train = np.load(os.path.join(root, "y_train.npy"))
x_test  = np.flip(np.load(os.path.join(root, "x_val.npy")), axis=1).copy()
y_test  = np.load(os.path.join(root, "y_val.npy"))

for dat, name in zip([x_train, y_train, x_test, y_test], ["x_train", "y_train", "x_test", "y_test"]):
    print(f"{name} shape: {dat.shape}")

# load original dataset
batch_size = 128
train_dataset = TokenizedTextClassificationDataset(x_train, y_train)
train_loader = DataLoader(
    train_dataset, shuffle=True, batch_size=batch_size
)
print(f"{len(train_dataset):>5,} training samples.")
val_dataset = TokenizedTextClassificationDataset(x_test, y_test)
val_loader = DataLoader(
    val_dataset, shuffle=True, batch_size=batch_size
)
print(f"{len(val_dataset):>5,} validation samples.")


n_epochs = 10
lr = 1.0
mu = 0.0
lam = 0.2
param_shapes = get_param_shapes(model.parameters())

with torch.no_grad():

    train_loss = torch.zeros(n_epochs)
    train_acc  = torch.zeros(n_epochs)
    val_loss   = torch.zeros(n_epochs)
    val_acc    = torch.zeros(n_epochs)

    torch.manual_seed(123)

    total_time = 0.0
    momentum = [torch.zeros(tuple(s)).to(DEVICE) for s in param_shapes]
    for epoch in range(n_epochs):
        tic = time.time()
        # if epoch > 0 and epoch % 5 == 0:
        #     print(f"epoch {epoch}: refitting gradient model...")
        #     with torch.enable_grad():
        #         gloss = train_one_epoch_grad(train_loader, model, gmodel, goptimizer, scores)
        #     print(f"epoch {epoch}: {gloss:0.6f} grad loss")
        
        for i, (idx, x, y) in enumerate(train_loader):
            model.zero_grad()
            with torch.enable_grad():
                logits = model(x.to(DEVICE))
                loss = F.binary_cross_entropy_with_logits(logits, y.to(DEVICE))
                sgrads = torch.autograd.grad(outputs=loss, inputs=model.parameters())
            grads = gmodel(flatten_parameters(sgrads), scores[idx])
            grads = unflatten_gradient(grads, param_shapes)

            for j, (p, g, m, s) in enumerate(zip(model.parameters(), grads, momentum, sgrads)):
                # g = (1 - lam) * s + lam * g
                # if epoch == 0:
                    # g = s # use regular sgd for the first epoch|
                # if epoch == 0:
                #     g = s # stochastic gradients
                # if j == 0 and i == 0:
                #     print(p)
                if mu > 0.0:
                    p.copy_(p - lr * (g + mu * m))
                    m.copy_(g)
                else:
                    p.copy_(p - lr * g)
        # train_one_epoch_func(train_loader, model, foptimizer)

        toc = time.time()
        total_time += (toc - tic)

        train_loss[epoch], train_acc[epoch] = evaluate_func(train_loader, model)
        val_loss[epoch], val_acc[epoch] = evaluate_func(val_loader, model)

        print(f"epoch {epoch}: train loss: {train_loss[epoch]:0.5f}")
        print(f"epoch {epoch}: train acc:  {train_acc[epoch]:0.5f}")
        print(f"epoch {epoch}: valid loss: {val_loss[epoch]:0.5f}")
        print(f"epoch {epoch}: valid acc:  {val_acc[epoch]:0.5f}")
        print()

torch.save(train_loss, "raoblackwell/sst2_rbsgd_train_loss.pt")
torch.save(train_acc, "raoblackwell/sst2_rbsgd_train_acc.pt")
torch.save(val_loss, "raoblackwell/sst2_rbsgd_val_loss.pt")
torch.save(val_acc, "raoblackwell/sst2_rbsgd_val_acc.pt")

print(total_time)
print(train_loss)
print(train_acc)
print(val_loss)
print(val_acc)