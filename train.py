import torch
import argparse

from src.experiment import ExperimentHelper
from src.image_data import get_cifar10_loaders

# TODO: ddp, gradient clipping, optimizer

# Option A: Use when running as a script.
# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "experiment_name",
#     type=str,
#     required=True,
#     help="name of experiment for entry in 'configs.py'",
# )
# parser.add_argument("seed", type=int, default=0, help="seed for entry in 'configs.py'")
# args = parser.parse_args()
# experiment_name, seed = args.experiment_name, args.seed

# Option B: Use when debugging.
experiment_name, seed = "debug", 0

# Build model and optimizer.
helper = ExperimentHelper(experiment_name, seed)
model = helper.get_model()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=helper.cfg["optim_cfg"]["lr"], weight_decay=5e-4
)

# TODO: Remove.
train_dataloader, test_dataloader = get_cifar10_loaders(512)
device = helper.device

# Run experiment.
model.train()
X, Y = helper.get_batch("train")
# for iter_num in range(helper.max_iters):
#     helper.log_step(iter_num, model)
#     for micro_step in range(helper.grad_accumulation_steps):
#         loss, logits = model(X, Y)
#         loss = loss / helper.grad_accumulation_steps
#         X, Y = helper.get_batch("train")
#         loss.backward()
#     optimizer.step()
#     optimizer.zero_grad(set_to_none=True)
iter_num = 0
for epoch in range(10):
    for X, Y in train_dataloader:
        helper.log_step(iter_num, model)
        for micro_step in range(helper.grad_accumulation_steps):
            loss, logits = model(X.to(device), Y.to(device))
            loss = loss / helper.grad_accumulation_steps
            # X, Y = helper.get_batch("train")
            loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        iter_num += 1
helper.end_experiment()
