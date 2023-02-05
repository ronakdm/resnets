import torch
import numpy as np

from data import get_cifar10_loaders
from models import MyrtleNet
from helper import ExperimentHelper, TRAIN, VAL

# Set configuration.
config = {
    "model_id" : "05",
    "n_layers" : 2,
    "residual_blocks" : [1],
    "n_epochs" : 24,
    "device" : "cuda:0",
    "model_name" : "myrtle_net",
    "experiment_name" : "loss_curve",
    "logs_dir" : "logs/",
    "output_dir" : "/mnt/hdd/ronak/cifar10_resnets",
    "batch_size" : 512,
    "lr" : 3e-4,
    "metrics" : ["loss", "accuracy"],
}

# Load model and data.
train_dataloader, test_dataloader = get_cifar10_loaders(config['batch_size'])
device = config['device']
model = MyrtleNet(
    n_layers=config['n_layers'], 
    residual_blocks=config['residual_blocks'],
).float().to(device)
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])

# Determine which metrics should be computed.
def evaluate(logits, labels):
    return {
        "loss": loss_func(logits, labels).item(),
        "accuracy": (torch.sum(torch.argmax(logits, dim=1) == labels) / len(labels)).item()
    }

# Initialize helper for logging and saving.
helper = ExperimentHelper(
    config, 
    len(train_dataloader), 
    len(test_dataloader), 
)

# Run experiment.
helper.start_experiment(model)
for epoch in range(config['n_epochs']):
    helper.start_epoch(epoch, TRAIN)
    for i, (x_batch, y_batch) in enumerate(train_dataloader):
        helper.start_step(i)

        model.zero_grad()
        logits = model(x_batch.to(device))
        loss = loss_func(logits, y_batch.to(device))
        loss.backward()
        optimizer.step()

        helper.end_step(evaluate(logits, y_batch.to(device)))
    helper.end_epoch(epoch)

    helper.start_epoch(epoch, VAL)
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(test_dataloader):
            helper.start_step(i)

            logits = model(x_batch.to(device))

            helper.end_step(evaluate(logits, y_batch.to(device)))
    helper.end_epoch(epoch, model=model)
helper.end_experiment()