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

TRAIN = "train"
VAL = "validation"


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


class ExperimentHelper:
    def __init__(self, experiment_name, seed):
        self.cfg = self._get_config(experiment_name, seed)

        # expose what is necessary
        self.max_iter = self.cfg["max_iter"]
        self.grad_accumulation_steps = self.cfg["grad_accumulation_steps"]

        # Record experiment parameters and metrics.
        self.max_iter = self.cfg["max_iter"]
        self.train_data, self.val_data = self._get_data(self.cfg.dataset)

        # Seed everything.
        seed = self.cfg["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Create logger and logging/output directories.
        save_dir = os.path.join(
            self.cfg["experiment_group"], self.cfg["experiment_name"], self.cfg["seed"]
        )
        self.log_dir = os.path.join(self.cfg["log_dir"], save_dir)
        self.output_dir = os.path.join(self.cfg["output_dir"], save_dir)
        os.makedirs(self.log_dir)
        os.makedirs(self.output_dir)
        with open(os.path.join(self.log_dir, "config.json"), "w") as outfile:
            json.dump(self.cfg, outfile, indent=4)
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(self.log_save_dir, "output.log")),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.best_val_loss = torch.inf
        self.epoch_stats = []
        self.total_t0 = time.time()
        self.t0 = time.time()
        logging.info(f"============================================")

    def _get_config(self, experiment_name, seed):
        for cfg in configs:
            if cfg["experiment_name"] == experiment_name and cfg["seed"] == seed:
                return cfg
        raise NotImplementedError(
            f"No configuration found for '{experiment_name}' with seed '{seed}'!"
        )

    def _get_data(self, dataset):
        pass

    def _format_time(self, elapsed):
        elapsed_rounded = int(round((elapsed)))
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def get_batch(self, split):
        # data = self.train_data if split == "train" else self.val_data
        # ix = torch.randint(len(data) - block_size, (batch_size,))
        # x = torch.stack(
        #     [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
        # )
        # y = torch.stack(
        #     [
        #         torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
        #         for i in ix
        #     ]
        # )
        # if self.device_type == "cuda":
        #     # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        #     x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(
        #         self.device, non_blocking=True
        #     )
        # else:
        #     x, y = x.to(self.device), y.to(self.device)
        # return x, y
        pass

    def get_model(self, model_cfg):
        arch = model_cfg["architecture"]
        if arch == "myrtle_net":
            model = MyrtleNet(**model_cfg)
        else:
            raise NotImplementedError(f"Unrecognized model architecture '{arch}'!")

        # Save a snapshot of the network architecture.
        with open(os.path.join(self.log_save_dir, "model.txt"), "w") as f:
            print(model, file=f)

        return model

    def log_step(self, iter_num, model):
        if iter_num % self.cfg["eval_interval"] == 0 and not iter_num == 0:
            print()
            logging.info(
                f"Steps {iter_num - self.cfg['eval_interval']:>5,} to {iter_num:>5,} took: {self._format_time(time.time() - self.t0)}."
            )
            logging.info(f"======== Step {iter_num + 1} / {self.max_iter} ========")
            logging.info("Training...")

            # Compute evaluation metrics.
            stats = self._compute_metrics(iter_num, model)
            with open(
                os.path.join(self.log_dir, f"step_{iter_num}.json"), "w"
            ) as outfile:
                json.dump(stats, outfile, indent=4)
            self.epoch_stats.append(stats)

            # Checkpoint model.
            if stats["val_loss"] < self.best_val_loss:
                self.best_val_loss = stats["val_loss"]
                torch.save(
                    model.state_dict(),
                    os.path.join(self.output_dir, f"ckpt_{iter_num}.pt"),
                )

            # Reset timer.
            self.t0 = time.time()

        elif iter_num % (self.cfg["eval_interval"] // 5) == 0 and not iter_num == 0:
            elapsed = format_time(time.time() - self.t0)
            logging.info(
                f"  step {iter_num:>5,} / {self.max_iter:>5,}.    elapsed: {elapsed}."
            )

    def _compute_metrics(self, iter_num, model):
        pass

    def end_experiment(self):
        print()
        logging.info(
            f"Training complete! Total time: {format_time(time.time() - self.total_t0)}"
        )

        # Save epoch metrics in readable format.
        df = pd.DataFrame(self.epoch_stats)
        with open(os.path.join(self.log_save_dir, "epoch_stats.csv"), "w") as f:
            df.to_csv(f, index=False)
