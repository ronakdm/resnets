#!/bin/bash

python train_sgdm.py --dataset imagenet_captions_50k --experiment_name clip_glove --device cuda:2 --seed=0
python train_sgdm.py --dataset imagenet_captions_50k --experiment_name clip_glove --device cuda:2 --seed=1
python train_sgdm.py --dataset imagenet_captions_50k --experiment_name clip_glove --device cuda:2 --seed=2
python train_sgdm.py --dataset imagenet_captions_50k --experiment_name clip_glove --device cuda:2 --seed=3

python train_sgdm.py --dataset imagenet_captions_50k --experiment_name joint_glove --device cuda:2 --seed=0
python train_sgdm.py --dataset imagenet_captions_50k --experiment_name joint_glove --device cuda:2 --seed=1
python train_sgdm.py --dataset imagenet_captions_50k --experiment_name joint_glove --device cuda:2 --seed=2
python train_sgdm.py --dataset imagenet_captions_50k --experiment_name joint_glove --device cuda:2 --seed=3

python train_sgdm.py --dataset imagenet_captions_50k --experiment_name double_glove --device cuda:2 --seed=0
python train_sgdm.py --dataset imagenet_captions_50k --experiment_name double_glove --device cuda:2 --seed=1
python train_sgdm.py --dataset imagenet_captions_50k --experiment_name double_glove --device cuda:2 --seed=2
python train_sgdm.py --dataset imagenet_captions_50k --experiment_name double_glove --device cuda:2 --seed=3