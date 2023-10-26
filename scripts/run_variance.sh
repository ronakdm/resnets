#!/bin/bash

# python train.py --experiment_name batch_size_8 --device cuda:0
# python train.py --experiment_name batch_size_32 --device cuda:0
# python train.py --experiment_name batch_size_128 --device cuda:0
# python train.py --experiment_name batch_size_512 --device cuda:0
# python train.py --experiment_name batch_size_2048 --device cuda:1
OMP_NUM_THREADS=3 torchrun --standalone --nproc_per_node=4 train.py --experiment_name batch_size_8192
# OMP_NUM_THREADS=3 torchrun --standalone --nproc_per_node=4 train.py --experiment_name batch_size_32768