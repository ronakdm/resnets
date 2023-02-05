import numpy as np
import torch
import random
import time
import datetime
import logging
import sys
import os
import pandas as pd

TRAIN = "train"
VAL = "validation"

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def get_save_dirs(experiment_name, model_name, experiment_id, logs_dir, output_dir):
    save_dirs = []
    for directory in [logs_dir, output_dir]:
        exp_path = os.path.join(directory, experiment_name)
        if not os.path.exists(exp_path):
            os.mkdir(exp_path)
        save_dir = os.path.join(exp_path, model_name + "_" + experiment_id)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_dirs.append(save_dir)
    return save_dirs[0], save_dirs[1]

class ExperimentHelper:
    def __init__(
        self, 
        n_epochs, 
        n_train_batches, 
        n_validation_batches, 
        metrics, 
        experiment_name,
        model_name,
        experiment_id,
        logs_dir,
        output_dir,
        seed=123
    ):
        self.n_epochs = n_epochs
        self.n_batches = {
            TRAIN: n_train_batches,
            VAL: n_validation_batches
        }
        self.seed = seed
        self.metrics = metrics
        self.log_save_dir, self.output_save_dir = get_save_dirs(experiment_name, model_name, experiment_id, logs_dir, output_dir)
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(self.log_save_dir, "output.log")),
                logging.StreamHandler(sys.stdout)
        ]
)

    def start_experiment(self, model):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        with open(os.path.join(self.log_save_dir, "model.txt"), 'w') as f:
            print(model, file=f)

        self.epoch_stats = []
        self.stats = {}
        self.total_t0 = time.time()

        logging.info(f"============================================")

    def start_epoch(self, epoch, mode):
        self.mode = mode
        print()
        if self.mode == TRAIN:
            logging.info(f"======== Epoch {epoch + 1} / {self.n_epochs} ========")
            logging.info("Training...")
            self.stats["epoch"] = epoch + 1
        elif self.mode == VAL:
            logging.info("Running validation...")
        self.t0 = time.time()
        self.total_train_loss = 0
        for metric in self.metrics:
            self.stats[mode + "_" + metric] = 0

    def start_step(self, it):
        if self.mode == TRAIN:
            if it % 40 == 0 and not it == 0:
                elapsed = format_time(time.time() - self.t0)
                logging.info(
                    f"  batch {it:>5,} / {self.n_batches[TRAIN]:>5,}.    elapsed: {elapsed}."
                )

    def end_step(self, eval_output):
        for metric in self.metrics:
            self.stats[self.mode + '_' + metric] += eval_output[metric]

    def end_epoch(self, epoch, model=None):
        for metric in self.metrics:
            logging.info(f"  {self.mode} {metric}: {self.stats[self.mode + '_' + metric] / self.n_batches[self.mode]:.3f}")
            self.stats[self.mode + '_' + metric] /= self.n_batches[self.mode]
        if self.mode == VAL:
            self.epoch_stats.append(self.stats.copy())
            if model:
                torch.save(model.state_dict(), os.path.join(self.output_save_dir, f"model_epoch_{epoch}.pt"))
        elapsed = format_time(time.time() - self.t0)
        print()
        logging.info(f"  {self.mode} epoch {epoch + 1} took: {elapsed}")

    def end_experiment(self):
        print()
        logging.info(
            f"Training complete! Total time: {format_time(time.time() - self.total_t0)}"
        )
        df = pd.DataFrame(self.epoch_stats)
        with open(os.path.join(self.log_save_dir, "epoch_stats.csv"), 'w') as f:
            df.to_csv(f, index=False)


