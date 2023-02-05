import numpy as np
import torch
import torch.nn as nn
import random
import time
import datetime

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def flat_accuracy(logits, labels):
    pred_flat = np.argmax(logits, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

class ExperimentHelper:
    def __init__(self, n_epochs, seed=123):
        self.n_epochs = n_epochs
        self.seed = seed
        self.training_stats = []
        self.loss_func = nn.CrossEntropyLoss()

    def start_experiment(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        self.total_t0 = time.time()

    def start_train_epoch(self, epoch):
        print()
        print(f"======== Epoch {epoch + 1} / {self.n_epochs} ========")
        print("Training...")
        self.t0 = time.time()
        self.total_train_loss = 0
        self.train_batches = 0

    def start_train_step(self, it):
        if it % 40 == 0 and not it == 0:
            elapsed = format_time(time.time() - self.t0)
            print(
                f"  Batch {it:>5,}  of  {self.n_batches:>5,}.    Elapsed: {elapsed}."
            )

    def end_train_step(self, loss):
        self.total_train_loss += loss.item()
        self.train_batches += 1

    def end_train_epoch(self):
        avg_train_loss = self.total_train_loss / self.train_batches
        training_time = format_time(time.time() - self.t0)

        print()
        print(f"  Average training loss: {avg_train_loss:.2f}")
        print(f"  Training epcoh took: {training_time}")

    def start_validation_epoch(self):
        print()
        print("Running Validation...")
        self.t0 = time.time()
        self.total_eval_accuracy = 0
        self.total_eval_loss = 0
        self.validation_batches = 0

    def start_validation_step(self):
        pass

    def end_validation_step(self, logits, labels):
        logits = logits.cpu().numpy()
        labels = labels.cpu().numpy()
        self.total_eval_loss += self.loss_func(logits, labels).item()
        self.total_eval_accuracy += flat_accuracy(logits, labels)
        self.validation_batches += 1

    def end_validation_epoch(self):
        avg_val_accuracy = self.total_eval_accuracy / self.validation_batches
        print(f"  Accuracy: {avg_val_accuracy:.2f}")
        avg_val_loss = self.total_eval_loss / self.validation_batches
        validation_time = format_time(time.time() - self.t0)
        print(f"  Validation Loss: {avg_val_loss:.2f}")
        print(f"  Validation took: {validation_time:}")

    def end_experiment(self):
        print()
        print("Training complete!")
        print(
            f"Total training took {format_time(time.time() - self.total_t0)} (h:mm:ss)"
        )


