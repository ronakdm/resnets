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
    "resnet_default": {
        "experiment_group": "resnet",
    },
    "resnet_raking": {"experiment_group": "resnet", "use_raking": True},
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
        "n_bins": 40,
        "augment": True,
        # model
        "model_cfg": {
            "architecture": "myrtle_net",
            "n_layers": 3,
            "residual_blocks": [0, 2],
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
        "n_bins": 40,
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
            "algo": "sgd",
            "lr": 0.05,
            "weight_decay": 5e-5,
            "momentum": 0.9,
        },
        "grad_accumulation_steps": 1,
        "num_raking_rounds": 2,
    },
}
