OMP_NUM_THREADS=3 torchrun --standalone --nproc_per_node=4 train.py --experiment_name debug_ddp