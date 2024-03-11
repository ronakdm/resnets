#!/bin/bash

python train_sgdm.py --dataset imagenet_captions_50k --experiment_name clip_glove_convnext --device cuda:3 --seed=0
python train_sgdm.py --dataset imagenet_captions_50k --experiment_name clip_glove_convnext --device cuda:3 --seed=1
python train_sgdm.py --dataset imagenet_captions_50k --experiment_name clip_glove_convnext --device cuda:3 --seed=2
python train_sgdm.py --dataset imagenet_captions_50k --experiment_name clip_glove_convnext --device cuda:3 --seed=3

python train_sgdm.py --dataset imagenet_captions_50k --experiment_name joint_glove_convnext --device cuda:3 --seed=0
python train_sgdm.py --dataset imagenet_captions_50k --experiment_name joint_glove_convnext --device cuda:3 --seed=1
python train_sgdm.py --dataset imagenet_captions_50k --experiment_name joint_glove_convnext --device cuda:3 --seed=2
python train_sgdm.py --dataset imagenet_captions_50k --experiment_name joint_glove_convnext --device cuda:3 --seed=3

python train_sgdm.py --dataset imagenet_captions_50k --experiment_name double_glove_convnext --device cuda:3 --seed=0
python train_sgdm.py --dataset imagenet_captions_50k --experiment_name double_glove_convnext --device cuda:3 --seed=1
python train_sgdm.py --dataset imagenet_captions_50k --experiment_name double_glove_convnext --device cuda:3 --seed=2
python train_sgdm.py --dataset imagenet_captions_50k --experiment_name double_glove_convnext --device cuda:3 --seed=3