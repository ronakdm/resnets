import numpy as np
import torch
import random
import time
import datetime
import logging
import sys
import os
import pandas as pd
import json
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam

from configs import configs, defaults
from src.image_models import MyrtleNet, ResNet
from src.image_data import get_cifar10_loaders


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


class ExperimentHelper:
    def __init__(self, experiment_name, seed, device):
        try:
            self.cfg = configs[experiment_name]
            default = defaults[self.cfg["experiment_group"]]
            for key in default:
                if key not in self.cfg:
                    self.cfg[key] = default[key]
        except KeyError:
            raise NotImplementedError(
                f"No configuration found for '{experiment_name}' with seed '{seed}'!"
            )

        # Expose what is necessary.
        self.max_iters = self.cfg["max_iters"]
        self.optim_cfg = self.cfg["optim_cfg"]
        (
            self.device,
            self.ddp,
            self.is_master_process,
            self.rank,
            self.world_size,
            self.accumulation_steps_per_device,
        ) = self._configure_ddp(device)
        self.rank = seed_offset = self.rank
        self.effective_batch_size = self.cfg["batch_size"]
        assert (
            self.effective_batch_size % self.accumulation_steps_per_device == 0
        ), "'grad_accumulation_steps' * 'world_size' must divide 'batch_size'"

        # Seed everything.
        random.seed(seed + seed_offset)
        np.random.seed(seed + seed_offset)
        torch.manual_seed(seed + seed_offset)
        torch.cuda.manual_seed_all(seed + seed_offset)

        # Create logger and logging/output directories.
        if self.is_master_process:
            save_dir = os.path.join(
                self.cfg["experiment_group"],
                experiment_name,
                str(seed),
            )
            self.log_dir = os.path.join(self.cfg["log_dir"], save_dir)
            self.output_dir = os.path.join(self.cfg["output_dir"], save_dir)
            os.makedirs(self.log_dir, exist_ok=True)
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.log_dir, "config.json"), "w") as outfile:
                json.dump(self.cfg, outfile, indent=4)
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s [%(levelname)s] %(message)s",
                handlers=[
                    logging.FileHandler(os.path.join(self.log_dir, "output.log")),
                    logging.StreamHandler(sys.stdout),
                ],
            )
            self.best_val_loss = torch.inf
            self.epoch_stats = []
            self.total_t0 = time.time()
            self.t0 = time.time()

    def _format_time(self, elapsed):
        elapsed_rounded = int(round((elapsed)))
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def _configure_ddp(self, device):
        ddp = int(os.environ.get("RANK", -1)) != -1
        grad_accumulation_steps = self.cfg["grad_accumulation_steps"]
        assert (
            grad_accumulation_steps == 1
        ), "Hand-coded optimizer currently does not support gradient accumulation!"
        if ddp:
            init_process_group(backend="nccl")
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            device = f"cuda:{local_rank}"
            torch.cuda.set_device(device)
            is_master_process = local_rank == 0
        else:
            is_master_process = True
            local_rank = 0
            world_size = 1
        assert (
            grad_accumulation_steps % world_size == 0
        ), "'world_size' must divide 'grad_accumulation_steps'"
        accumulation_steps_per_device = grad_accumulation_steps // world_size

        return (
            device,
            ddp,
            is_master_process,
            local_rank,
            world_size,
            accumulation_steps_per_device,
        )

    def get_dataloaders(self, batch_size, rank):
        dataset = self.cfg["dataset"]
        root = self.cfg["data_dir"]

        if dataset == "cifar10":
            return get_cifar10_loaders(batch_size, rank, root=root)
        raise NotImplementedError(f"Unrecognized dataset '{dataset}'!")

    def get_model(self):
        model_cfg = self.cfg["model_cfg"]
        arch = model_cfg["architecture"]
        if arch == "myrtle_net":
            model = MyrtleNet(**model_cfg).float()
        elif arch == "resnet":
            model = ResNet(**model_cfg).float()
        else:
            raise NotImplementedError(f"Unrecognized model architecture '{arch}'!")

        if isinstance(self.cfg["init_from"], int):
            # attempt to resume from a checkpoint.
            iter_num = self.cfg["init_from"]
            model.load_state_dict(
                torch.load(os.path.join(self.output_dir, f"ckpt_{iter_num}.pt"))
            )

        # Save a snapshot of the network architecture.
        if self.is_master_process:
            with open(os.path.join(self.log_dir, "model.txt"), "w") as f:
                print(model, file=f)
        model.to(self.device)

        if self.ddp:
            model = DDP(model, device_ids=[self.device])

        return model

    def get_optimizer(self, model):
        optim_cfg = self.cfg["optim_cfg"]
        algo = optim_cfg["algo"]
        optimizers = {"sgd": SGD, "adam": Adam}
        del optim_cfg["algo"]
        try:
            optimizer = optimizers[algo](model.parameters(), **optim_cfg)
            # scheduler = torch.optim.lr_scheduler.OneCycleLR(
            #     optimizer,
            #     max_lr=optim_cfg["lr"] * 10,
            #     final_div_factor=10,
            #     steps_per_epoch=self.cfg["eval_interval"],
            #     total_steps=self.max_iters,
            #     pct_start=0.05,
            # )
        except KeyError:
            raise NotImplementedError(f"Unrecognized optimization algorithm '{algo}'!")
        # return optimizer, scheduler
        return optimizer

    def log_step(self, macro_iter_num, model, loaders):
        if (
            self.is_master_process
            and macro_iter_num % self.accumulation_steps_per_device == 0
        ):
            iter_num = macro_iter_num // self.accumulation_steps_per_device
            if iter_num % self.cfg["eval_interval"] == 0:
                if not iter_num == 0:
                    print()
                    logging.info(
                        f"Steps {iter_num - self.cfg['eval_interval']:>5,} to {iter_num:>5,} took: {self._format_time(time.time() - self.t0)}."
                    )
                    print()

                    logging.info(
                        f"Evaluating using {self.cfg['eval_iters']} batches..."
                    )
                    self.t0 = time.time()
                    # Compute evaluation metrics.
                    stats = self._compute_metrics(iter_num, model, loaders)
                    with open(
                        os.path.join(self.log_dir, f"step_{iter_num}.json"), "w"
                    ) as outfile:
                        json.dump(stats, outfile, indent=4)
                    self.epoch_stats.append(stats)
                    for metric in stats:
                        logging.info(f"    {metric}: {stats[metric]:0.4f}")
                    logging.info(
                        f"Evaluation took: {self._format_time(time.time() - self.t0)}."
                    )
                    # Checkpoint model.
                    if stats["validation_loss"] < self.best_val_loss:
                        logging.info(f"Saving checkpoint to '{self.output_dir}'...")
                        self.best_val_loss = stats["validation_loss"]
                        raw_model = model if not self.ddp else model.module
                        torch.save(
                            raw_model.state_dict(),
                            os.path.join(self.output_dir, f"ckpt_{iter_num}.pt"),
                        )

                if not iter_num == self.max_iters:
                    print()
                    logging.info(
                        f"======== Step {iter_num + 1:>5,} / {self.max_iters:>5,} ========"
                    )
                    logging.info("Training...")

                # Reset timer.
                self.t0 = time.time()

            elif iter_num % (self.cfg["eval_interval"] // 5) == 0 and not iter_num == 0:
                elapsed = format_time(time.time() - self.t0)
                logging.info(
                    f"    step {iter_num:>5,} / {self.max_iters:>5,}.    elapsed: {elapsed}."
                )

    @torch.no_grad
    def _compute_metrics(self, iter_num, model, loaders):
        # TODO: Make this work beyond image classification.
        out = {"iter_num": iter_num}
        model.eval()
        eval_iters = self.cfg["eval_iters"]
        out = {}
        for split, loader in zip(["train", "validation"], loaders):
            it = 0
            for X, Y in loader:
                Y = Y.to(self.device)
                loss, logits = model(X.to(self.device), Y)
                out[f"{split}_accuracy"] = (
                    torch.sum((torch.argmax(logits, dim=1) == Y)) / len(Y)
                ).item()
                out[f"{split}_loss"] = loss.item()
                it += 1
                if it > eval_iters:
                    break
        model.train()
        return out

    def end_experiment(self):
        if self.is_master_process:
            print()
            logging.info(
                f"Training complete! Total time: {format_time(time.time() - self.total_t0)}"
            )

            # Save epoch metrics in readable format.
            df = pd.DataFrame(self.epoch_stats)
            with open(os.path.join(self.log_dir, "epoch_stats.csv"), "w") as f:
                df.to_csv(f, index=False)
        if self.ddp:
            destroy_process_group()
