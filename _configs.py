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
    "batch_size_8": {
        "experiment_group": "variance",
        "eval_iters": 640,
        "batch_size": 8,
        "grad_accumulation_steps": 1,
    },
    "batch_size_32": {
        "experiment_group": "variance",
        "eval_iters": 160,
        "batch_size": 32,
        "grad_accumulation_steps": 1,
    },
    "batch_size_128": {
        "experiment_group": "variance",
        "eval_iters": 40,
        "batch_size": 128,
        "grad_accumulation_steps": 1,
    },
    "batch_size_512": {
        "experiment_group": "variance",
        "eval_iters": 10,
        "batch_size": 512,
        "grad_accumulation_steps": 1,
    },
    "batch_size_2048": {
        "experiment_group": "variance",
        "eval_iters": 5,
        "batch_size": 2048,
        "grad_accumulation_steps": 4,
    },
    "b8": {
        "experiment_group": "variance",
        "eval_iters": 640,
        "batch_size": 8,
        "grad_accumulation_steps": 1,
        "augment": False,
    },
    "b32": {
        "experiment_group": "variance",
        "eval_iters": 160,
        "batch_size": 32,
        "grad_accumulation_steps": 1,
        "augment": False,
    },
    "b128": {
        "experiment_group": "variance",
        "eval_iters": 40,
        "batch_size": 128,
        "grad_accumulation_steps": 1,
        "augment": False,
    },
    "b512": {
        "experiment_group": "variance",
        "eval_iters": 10,
        "batch_size": 512,
        "grad_accumulation_steps": 1,
        "augment": False,
    },
    "b2048": {
        "experiment_group": "variance",
        "eval_iters": 5,
        "batch_size": 2048,
        "grad_accumulation_steps": 4,
        "augment": False,
    },
    "batch_size_8192": {
        "experiment_group": "variance",
        "eval_iters": 5,
        "batch_size": 8192,
        "grad_accumulation_steps": 8,
    },
    "batch_size_32768": {
        "experiment_group": "variance",
        "eval_iters": 5,
        "batch_size": 32768,
        "grad_accumulation_steps": 16,
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
    "variance": {
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
        "max_iters": 3000,
        "eval_interval": 100,
        "eval_iters": 5,
        "batch_size": 32,
        "use_raking": False,
        "optim_cfg": {
            "algo": "sgd",
            "lr": 0.003,
            "weight_decay": 0.0005,
            "momentum": 0.9,
        },
        "grad_accumulation_steps": 16,
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
