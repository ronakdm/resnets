#!/bin/bash

# python train_sgdm.py --dataset=fashion_mnist --experiment_name raking_r2_k50_b64_u10 --device cuda:3 --seed=1
# python train_sgdm.py --dataset=fashion_mnist --experiment_name raking_r2_k50_b64_u10 --device cuda:3 --seed=2
# python train_sgdm.py --dataset=fashion_mnist --experiment_name raking_r2_k50_b64_u10 --device cuda:3 --seed=3
# python train_sgdm.py --dataset=fashion_mnist --experiment_name raking_r2_k50_b64_u10 --device cuda:3 --seed=4
# python train_sgdm.py --dataset=fashion_mnist --experiment_name raking_r2_k50_b64_u10 --device cuda:3 --seed=5
python train_sgdm.py --dataset ub_fmnist --experiment_name raking_r2_k50_b256 --device cuda:3 --seed=2