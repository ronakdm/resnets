#!/bin/bash

# python train_sgdm.py --dataset=cifar10 --experiment_name raking_r2_k50_b128_u10 --device cuda:0 --seed=1
# python train_sgdm.py --dataset=cifar10 --experiment_name raking_r2_k50_b128_u10 --device cuda:0 --seed=2
# python train_sgdm.py --dataset=cifar10 --experiment_name raking_r2_k50_b128_u10 --device cuda:0 --seed=3
# python train_sgdm.py --dataset=cifar10 --experiment_name raking_r2_k50_b128_u10 --device cuda:0 --seed=4
# python train_sgdm.py --dataset=cifar10 --experiment_name raking_r2_k50_b128_u10 --device cuda:0 --seed=5
python train_sgdm.py --dataset ub_fmnist --experiment_name default_b256 --device cuda:0 --seed=1