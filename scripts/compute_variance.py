import torch
import numpy as np
import os
import json
import sys
from tqdm import tqdm
import argparse

sys.path.extend(["..", "."])
from src.image_data import get_quantized_cifar10_loaders
from src.image_models import MyrtleNet, ResNet


def compute_variance(model, mean_loader, variance_loader, sims, device="cpu"):
    m = 20
    for i, (x, y, cx, cy) in enumerate(mean_loader):
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
        for x, y, cx, cy in variance_loader:
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


def compute_variance_curve(
    exp_name, exp_group, seed=0, sims=50, device="cuda:3", augment=True
):
    out_dir = f"/mnt/ssd/ronak/output/{exp_group}/{exp_name}/{seed}/"
    log_dir = f"/home/ronak/resnets/logs/{exp_group}/{exp_name}/{seed}/"
    steps = torch.arange(100, 3100, 100)

    with open(os.path.join(log_dir, "config.json"), "r") as infile:
        config = json.load(infile)
    model_cfg = config["model_cfg"]

    mean_loader, _, _ = get_quantized_cifar10_loaders(1024, 0)
    variance_loader, _, _ = get_quantized_cifar10_loaders(512, 0)

    variances = []
    observed_steps = []
    for step in tqdm(steps):
        fname = os.path.join(out_dir, f"ckpt_{step}.pt")
        if os.path.exists(fname):
            model = ResNet(**model_cfg)
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
        f"notebooks/output/{exp_name}_steps_seed_{seed}.npy", np.array(observed_steps)
    )
    np.save(
        f"notebooks/output/{exp_name}_variances_seed_{seed}.npy", np.array(variances)
    )


# parser = argparse.ArgumentParser()
# parser.add_argument("--device", type=str, required=True, help="gpu index")
# parser.add_argument("--seed", type=str, required=True, help="experiment seed")
# args = parser.parse_args()
# seed, device = args.seed, args.device

seed, device = 0, "cuda:2"

exp_group = "resnet"
for exp_name in ["resnet_raking"]:
    print(f"Computing variance for {exp_name }...")
    compute_variance_curve(
        exp_name, exp_group, seed=seed, sims=50, device=device, augment=False
    )
