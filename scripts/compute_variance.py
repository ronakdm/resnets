import torch
import numpy as np
import os
import json
import sys
from tqdm import tqdm
import argparse

sys.path.extend(["..", "."])
from src.image_data import get_cifar10_loaders
from src.image_models import MyrtleNet


def compute_variance(model, mean_loader, variance_loader, sims, device="cpu"):
    # compute mean using accumulation
    m = 20
    for i, (x, y) in enumerate(mean_loader):
        if i >= m:
            break
        loss, logits = model(x.to(device), y.to(device))
        loss /= m
        loss.backward()
    mean = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            mean[name] = param.grad

    iter_num = 0
    variance = 0
    while iter_num < sims:
        for x, y in variance_loader:
            if iter_num >= sims:
                break

            model.zero_grad()
            loss, logits = model(x.to(device), y.to(device))
            loss.backward()
            for name, param in model.named_parameters():
                if param.requires_grad:
                    variance += torch.norm(param.grad - mean[name]) ** 2 / sims

            iter_num += 1
    return variance.detach().cpu().item()


def compute_variance_curve(batch_size, seed=0, sims=50, device="cuda:3"):
    np.save(f"notebooks/output/test.npy", np.ones((5)))

    out_dir = f"/mnt/ssd/ronak/output/variance/batch_size_{batch_size}/{seed}/"
    log_dir = f"/home/ronak/resnets/logs/variance/batch_size_{batch_size}/{seed}/"
    steps = torch.arange(100, 3100, 100)

    with open(os.path.join(log_dir, "config.json"), "r") as infile:
        config = json.load(infile)
    model_cfg = config["model_cfg"]

    mean_loader, _ = get_cifar10_loaders(1024, 0)
    variance_loader, _ = get_cifar10_loaders(batch_size, 0)

    variances = []
    observed_steps = []
    for step in tqdm(steps):
        fname = os.path.join(out_dir, f"ckpt_{step}.pt")
        if os.path.exists(fname):
            model = MyrtleNet(**model_cfg)
            model.load_state_dict(torch.load(fname))
            model.to(device)
            model.eval()
            observed_steps.append(step.item())
            variances.append(
                compute_variance(
                    model, mean_loader, variance_loader, sims, device=device
                )
            )

    np.save(
        f"notebooks/output/batch_size_{batch_size}_steps_seed_{seed}.npy",
        np.array(observed_steps),
    )
    np.save(
        f"notebooks/output/batch_size_{batch_size}_variances_seed_{seed}.npy",
        np.array(variances),
    )


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, required=True, help="gpu index")
parser.add_argument("--seed", type=str, required=True, help="experiment seed")
args = parser.parse_args()


for batch_size in [2048]:
    print(f"Computing variance for batch size {batch_size}...")
    compute_variance_curve(batch_size, seed=args.seed, sims=50, device=args.device)
