#!/bin/bash

python train_sgdm.py --dataset imagenet_captions_50k --experiment_name clip_glove --device cuda:2 --seed=0
# python train_sgdm.py --dataset imagenet_captions_50k --experiment_name jointly_centered --device cuda:2 --seed=2
# python train_sgdm.py --dataset imagenet_captions_50k --experiment_name doubly_centered --device cuda:2 --seed=2