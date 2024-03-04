#!/bin/bash

python train_sgdm.py --dataset imagenet_captions_50k --experiment_name default --device cuda:3 --seed=3
python train_sgdm.py --dataset imagenet_captions_50k --experiment_name jointly_centered --device cuda:3 --seed=3
python train_sgdm.py --dataset imagenet_captions_50k --experiment_name doubly_centered --device cuda:3 --seed=3