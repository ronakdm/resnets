#!/bin/bash

model="doubly_centered"

cd /mnt/ssd/ronak/output/imagenet_captions_50k/$model
for dataset in cifar10 cifar100 stl10 caltech101
do
    clip_benchmark eval --dataset=$dataset --task=zeroshot_classification --pretrained='laion2b_s34b_b79k' --model=$model --model_type=miniclip --output=$dataset.json --batch_size=64
done                    