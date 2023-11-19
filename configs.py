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
    "resnet_default": {
        "experiment_group": "resnet",
    },
    "resnet_raking": {"experiment_group": "resnet", "use_raking": True},
    "resnet_raking_rounds_4": {"experiment_group": "resnet", "use_raking": True, "num_raking_rounds": 4},
    "resnet_raking_bins_25": {"experiment_group": "resnet", "use_raking": True, "n_bins": 25},
    "resnet_raking_bins_100": {"experiment_group": "resnet", "use_raking": True, "n_bins": 100},
    "resnet_raking_bins_200": {"experiment_group": "resnet", "use_raking": True, "n_bins": 200},
    "resnet_default_batch_128": {"experiment_group": "resnet", "batch_size": 128, "eval_iters": 20, "max_iters": 6000, "eval_interval": 200},
    "resnet_raking_batch_128": {"experiment_group": "resnet", "batch_size": 128, "use_raking": True, "eval_iters": 20, "max_iters": 6000, "eval_interval": 200},
    "resnet_default_batch_32": {"experiment_group": "resnet", "batch_size": 32, "eval_iters": 80, "max_iters": 10000, "eval_interval": 400},
    "resnet_raking_batch_32": {"experiment_group": "resnet", "batch_size": 32, "use_raking": True, "eval_iters": 80, "max_iters": 10000, "eval_interval": 400},
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
        "augment": True,
        # model
        "model_cfg": {
            "architecture": "resnet",
            "n_layers": 5,
        },
        "init_from": "scratch",
        # taining
        "max_iters": 3000,
        "eval_interval": 100,
        "eval_iters": 5,
        "batch_size": 512,
        "use_raking": False,
        "optim_cfg": {
            "algo": "adam",
            "lr": 1e-3,
            # "weight_decay": 1e-3
        },
        "grad_accumulation_steps": 1,
        "num_raking_rounds": 2,
    },
}
