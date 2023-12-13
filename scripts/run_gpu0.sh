#!/bin/bash

python train_sgdm.py --dataset=cifar10 --experiment_name default_b128_u10 --device cuda:0 --seed=1
python train_sgdm.py --dataset=cifar10 --experiment_name default_b128_u10 --device cuda:0 --seed=2
python train_sgdm.py --dataset=cifar10 --experiment_name default_b128_u10 --device cuda:0 --seed=3
python train_sgdm.py --dataset=cifar10 --experiment_name default_b128_u10 --device cuda:0 --seed=4
python train_sgdm.py --dataset=cifar10 --experiment_name default_b128_u10 --device cuda:0 --seed=5