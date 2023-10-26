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
from configs import configs

from src.models import MyrtleNet
from src.image_data import get_cifar10_loaders


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


class ExperimentHelper:
    def __init__(self, experiment_name, seed):
        self.cfg = self._get_config(experiment_name, seed)

        # Expose what is necessary.
        self.max_iters = self.cfg["max_iters"]
        self.grad_accumulation_steps = self.cfg["grad_accumulation_steps"]
        self.device = self.cfg["device"]

        # Load datasets.
        # self.train_loader, self.val_loader, self.metric_func = self._load_dataset(
        #     self.cfg["dataset"]
        # )

        # Seed everything.
        seed = self.cfg["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Create logger and logging/output directories.
        save_dir = os.path.join(
            self.cfg["experiment_group"],
            self.cfg["experiment_name"],
            str(self.cfg["seed"]),
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

    def _get_config(self, experiment_name, seed):
        for cfg in configs:
            if cfg["experiment_name"] == experiment_name and cfg["seed"] == seed:
                return cfg
        raise NotImplementedError(
            f"No configuration found for '{experiment_name}' with seed '{seed}'!"
        )

    # def _load_dataset(self, dataset):
    #     if dataset == "cifar10":
    #         return load_cifar10()
    #     raise NotImplementedError(f"Unrecognized dataset '{dataset}'!")

    def _format_time(self, elapsed):
        elapsed_rounded = int(round((elapsed)))
        return str(datetime.timedelta(seconds=elapsed_rounded))

    # def get_batch(self, split):
    #     loader = self.train_loader if split == "train" else self.val_loader
    #     return loader.get_batch(self.cfg["batch_size"], self.device)

    def get_dataloaders(self):
        dataset = self.cfg["dataset"]
        batch_size = self.cfg["batch_size"]
        root = self.cfg["data_dir"]

        if dataset == "cifar10":
            return get_cifar10_loaders(batch_size, root=root)
        raise NotImplementedError(f"Unrecognized dataset '{dataset}'!")

    def get_model(self):
        model_cfg = self.cfg["model_cfg"]
        arch = model_cfg["architecture"]
        if arch == "myrtle_net":
            model = MyrtleNet(**model_cfg).float()
        else:
            raise NotImplementedError(f"Unrecognized model architecture '{arch}'!")

        # Save a snapshot of the network architecture.
        with open(os.path.join(self.log_dir, "model.txt"), "w") as f:
            print(model, file=f)

        model.to(self.device)
        return model

    def log_step(self, iter_num, model, loaders):
        if iter_num % self.cfg["eval_interval"] == 0:
            if not iter_num == 0:
                print()
                logging.info(
                    f"Steps {iter_num - self.cfg['eval_interval']:>5,} to {iter_num:>5,} took: {self._format_time(time.time() - self.t0)}."
                )
                print()

                logging.info(f"Evaluating using {self.cfg['eval_iters']} batches...")
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
                    torch.save(
                        model.state_dict(),
                        os.path.join(self.output_dir, f"ckpt_{iter_num}.pt"),
                    )

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
        print()
        logging.info(
            f"Training complete! Total time: {format_time(time.time() - self.total_t0)}"
        )

        # Save epoch metrics in readable format.
        df = pd.DataFrame(self.epoch_stats)
        with open(os.path.join(self.log_dir, "epoch_stats.csv"), "w") as f:
            df.to_csv(f, index=False)
