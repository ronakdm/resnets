import torch
import argparse

from src.experiment import ExperimentHelper
from src.raking import get_raking_ratio_weight

# TODO: gradient clipping 

# Option A: Use when running as a script.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--experiment_name",
    type=str,
    required=True,
    help="name of experiment for entry in 'configs.py'",
)
parser.add_argument(
    "--seed", type=int, default=0, help="seed for entry in 'configs.py'"
)
parser.add_argument("--device", type=str, default="cuda:0", help="gpu index")
args = parser.parse_args()
experiment_name, seed, device = args.experiment_name, args.seed, args.device

# Option B: Use with debugger.
# experiment_name, seed, device = "sst2_raking_batch_128", 0, "cuda:0"

# Build model.
helper = ExperimentHelper(experiment_name, seed, device)
model = helper.get_model()

# Configure distributed data parallel.
is_ddp_run, device, rank, world_size = (
    helper.ddp,
    helper.device,
    helper.rank,
    helper.world_size,
)

# Load data.
accumulation_steps_per_device = helper.accumulation_steps_per_device
batch_size = helper.effective_batch_size // (accumulation_steps_per_device * world_size)
train_loader, val_loader, quantization = helper.get_dataloaders(batch_size, rank)
use_raking = helper.use_raking
num_rounds = helper.num_raking_rounds

# Build optimizer.
optimizer = helper.get_optimizer(model)

# Run experiment.
model.train()
momentum = [torch.zeros(param.shape).to(device) for param in model.parameters()]
iter_num = 0
while iter_num < helper.max_iters * accumulation_steps_per_device:
    for idx, X, Y in train_loader:
        iter_num += 1
        helper.log_step(iter_num, model, [train_loader, val_loader])
        if iter_num >= helper.max_iters:
            break

        if is_ddp_run:
            model.require_backward_grad_sync = (
                iter_num % accumulation_steps_per_device == 0
            )

        if use_raking:
            sample_weight = torch.from_numpy(get_raking_ratio_weight(
                idx, quantization, num_rounds
            )).float().to(device=device, non_blocking=True)
            loss, logits = model(X.to(device), Y.to(device), sample_weight=sample_weight.to(device))
        else:
            loss, logits = model(X.to(device), Y.to(device))
        loss = loss / accumulation_steps_per_device
        loss.backward()

        if iter_num % accumulation_steps_per_device == 0:
            lr = helper.get_lr(iter_num)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad(set_to_none=True)
helper.end_experiment()
