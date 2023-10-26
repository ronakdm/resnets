import torch
import argparse

from src.experiment import ExperimentHelper

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

train_loader, val_loader = helper.get_dataloaders()
device = helper.device

# Run experiment.
model.train()
iter_num = 0
while iter_num < helper.max_iters:
    for X, Y in train_loader:
        helper.log_step(iter_num, model, [train_loader, val_loader])
        loss, logits = model(X.to(device), Y.to(device))
        loss = loss / helper.grad_accumulation_steps
        loss.backward()
        if iter_num % helper.grad_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        iter_num += 1
        if iter_num > helper.max_iters:
            break
helper.end_experiment()
