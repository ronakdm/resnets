#!/bin/bash

model="ViT-B-32"
model_type="open_clip"

cd /mnt/ssd/ronak/output/imagenet_captions_50k/$model
for dataset in cifar10 cifar100 stl10
do
    clip_benchmark eval --dataset=$dataset --task=zeroshot_classification --pretrained='laion2b_s34b_b79k' --model=$model --model_type=$model_type --output=$dataset.json --batch_size=64
done                    