configs = {
    "debug": {
        # experiment
        "experiment_group": "debug",
    },
    "debug_ddp": {
        # experiment
        "experiment_group": "debug",
        # model
        "model_cfg": {
            "architecture": "myrtle_net",
            "n_layers": 1,
            "residual_blocks": [],
        },
        # taining
        "max_iters": 200,
        "eval_interval": 100,
        "eval_iters": 5,
        "batch_size": 50000,
        "grad_accumulation_steps": 40,
    },
    "debug_resnet": {
        # experiment
        "experiment_group": "debug",
        # model
        "model_cfg": {
            "architecture": "resnet",
            "n_layers": 3,
        },
        # taining
        "max_iters": 300,
        "eval_interval": 100,
        "eval_iters": 5,
        "batch_size": 512,
        "grad_accumulation_steps": 1,
    },
    "debug_raking": {"experiment_group": "debug", "use_raking": True},
    "resnet_default_batch_128": {"experiment_group": "resnet"},
    "resnet_raking_batch_128":  {"experiment_group": "resnet", "use_raking": True},
    "resnet_raking_rounds_4": {"experiment_group": "resnet", "use_raking": True, "num_raking_rounds": 4},
    "resnet_raking_rounds_6": {"experiment_group": "resnet", "use_raking": True, "num_raking_rounds": 6},
    "resnet_default_factor_5": {"experiment_group": "resnet", "factor": 5.0},
    "resnet_raking_factor_5": {"experiment_group": "resnet", "use_raking": True, "factor": 5.0},
    "resnet_default_factor_10": {"experiment_group": "resnet", "factor": 10.0},
    "resnet_raking_factor_10": {"experiment_group": "resnet", "use_raking": True, "factor": 10.0},
    "resnet_raking_bins_25":  {"experiment_group": "resnet", "use_raking": True, "n_bins": 25},
    "resnet_raking_bins_100": {"experiment_group": "resnet", "use_raking": True, "n_bins": 100},
    "resnet_raking_bins_200": {"experiment_group": "resnet", "use_raking": True, "n_bins": 200},
    "resnet_default_batch_512": {"experiment_group": "resnet", "batch_size": 512, "eval_iters": 20, "max_iters": 6000, "eval_interval": 200},
    "resnet_raking_batch_512":  {"experiment_group": "resnet", "batch_size": 512, "use_raking": True, "eval_iters": 20, "max_iters": 6000, "eval_interval": 200},
    "resnet_default_batch_32":  {"experiment_group": "resnet", "batch_size": 32, "eval_iters": 80, "max_iters": 10000, "eval_interval": 400},
    "resnet_raking_batch_32":   {"experiment_group": "resnet", "batch_size": 32, "use_raking": True, "eval_iters": 80, "max_iters": 10000, "eval_interval": 400},
    # fashion
    "fmnist_default_batch_128":  {"experiment_group": "resnet", "dataset": "fashion_mnist"},
    "fmnist_raking_batch_128":  {"experiment_group": "resnet", "dataset": "fashion_mnist", "use_raking": True},
    "fmnist_default_factor_5":  {"experiment_group": "resnet", "dataset": "fashion_mnist", "factor": 5.0},
    "fmnist_raking_factor_5":  {"experiment_group": "resnet", "dataset": "fashion_mnist",  "use_raking": True,  "factor": 5.0},
    "fmnist_default_factor_10":  {"experiment_group": "resnet", "dataset": "fashion_mnist", "factor": 10.0},
    "fmnist_raking_factor_10":  {"experiment_group": "resnet", "dataset": "fashion_mnist", "use_raking": True,  "factor": 10.0},
    "fmnist_default_factor_100":  {"experiment_group": "resnet", "dataset": "fashion_mnist", "factor": 100.0},
    "fmnist_raking_factor_100":  {"experiment_group": "resnet", "dataset": "fashion_mnist", "use_raking": True,  "factor": 100.0},
    # sst2
    "sst2_default_batch_128":  {"experiment_group": "sst2"},
    "sst2_raking_batch_128":  {"experiment_group": "sst2", "use_raking": True},
    "sst2_default_batch_32":  {"experiment_group": "sst2", "batch_size": 32},
    "sst2_raking_batch_32":  {"experiment_group": "sst2", "batch_size": 32, "use_raking": True},
    "sst2_default_factor_10":  {"experiment_group": "sst2", "batch_size": 128, "factor": 10.0},
    "sst2_raking_factor_10":  {"experiment_group": "sst2", "batch_size": 128, "use_raking": True, "factor": 10.0},
}

model_cfgs = {
    "fashion_mnist": {
        "architecture": "resnet",
        "n_layers": 2,
        "n_classes": 10,
        "height": 28,
        "width": 28,
    },
    "cifar10":  {
        "architecture": "resnet",
        "n_layers": 3,
        "n_classes": 10,
        "height": 32,
        "width": 32,
    },
    "sst2":  {
        "architecture": "transformer",
        "block_size": 77,
        "vocab_size": 49408,
        "n_layer": 2,
        "n_head": 8,
        "n_embd": 64,
        "n_class": 2,
        "dropout": 0.0,
        "bias": False,
    },
}

# maps experiment group to default settings
defaults = {
    "debug": {
        # experiment
        "log_dir": "logs/",
        "output_dir": "/mnt/ssd/ronak/output",
        # data
        "data_dir": "/mnt/ssd/ronak/datasets/",
        "dataset": "cifar10",
        "n_bins": 50,
        "augment": True,
        # model
        "model_cfg": {
            "architecture": "resnet",
            "n_layers": 3,
        },
        "init_from": "scratch",
        # taining
        "max_iters": 300,
        "eval_interval": 100,
        "eval_iters": 5,
        "batch_size": 512,
        "use_raking": False,
        "optim_cfg": {
            "algo": "sgd",
            "lr": 0.003,
            "weight_decay": 5e-4,
            "momentum": 0.9,
        },
        "grad_accumulation_steps": 1,
        "num_raking_rounds": 2,
    },
    "resnet": {
        # experiment
        "log_dir": "logs/",
        "output_dir": "/mnt/ssd/ronak/output",
        # data
        "data_dir": "/mnt/ssd/ronak/datasets/",
        "dataset": "cifar10",
        "n_bins": 50,
        "factor": 1.0,
        # model
        # "model_cfg": {
        #     "architecture": "resnet",
        #     "n_layers": 5,
        # },
        "init_from": "scratch",
        # taining
        "max_iters": 4000,
        "eval_interval": 200,
        "eval_iters": 200,
        "batch_size": 128,
        "use_raking": False,
        # "optim_cfg": {
        #     "algo": "adam",
        #     "lr": 1e-3,
        #     # "weight_decay": 1e-3
        # },
        "optim_cfg": {
            "algo": "sgd",
            "lr": 1e-3,
            "momentum": 0.5
        },
        "grad_accumulation_steps": 1,
        "num_raking_rounds": 2,
    },
    "sst2": {
        # experiment
        "log_dir": "logs/",
        "output_dir": "/mnt/ssd/ronak/output",
        # data
        "data_dir": "/mnt/ssd/ronak/datasets/",
        "dataset": "sst2",
        "n_bins": 50,
        "factor": 1.0,
        "init_from": "scratch",
        # taining
        "max_iters": 4000,
        "eval_interval": 200,
        "eval_iters": 100,
        "batch_size": 128,
        "use_raking": False,
        "optim_cfg": {
            "algo": "adam",
            "lr": 1e-3,
            "weight_decay": 1e-4
        },
        # "optim_cfg": {
        #     "algo": "sgd",
        #     "lr": 1e-3,
        #     # "momentum": 0.0
        # },
        "grad_accumulation_steps": 1,
        "num_raking_rounds": 2,
    },
}
