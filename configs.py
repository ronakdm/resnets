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
        # experiment
        "experiment_group": "variance",
        "eval_iters": 40,
        "batch_size": 128,
        "grad_accumulation_steps": 1,
    },
    "batch_size_512": {
        # experiment
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
        "optim_cfg": {
            "algo": "sgd",
            "lr": 0.003,
            "weight_decay": 5e-4,
            "momentum": 0.9,
        },
        "grad_accumulation_steps": 1,
    },
    "variance": {
        # experiment
        "log_dir": "logs/",
        "output_dir": "/mnt/ssd/ronak/output",
        # data
        "data_dir": "/mnt/ssd/ronak/datasets/",
        "dataset": "cifar10",
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
        "optim_cfg": {
            "algo": "sgd",
            "lr": 0.003,
            "weight_decay": 0.0005,
            "momentum": 0.9,
        },
        "grad_accumulation_steps": 16,
    },
    "resnet": {
        # experiment
        "log_dir": "logs/",
        "output_dir": "/mnt/ssd/ronak/output",
        # data
        "data_dir": "/mnt/ssd/ronak/datasets/",
        "dataset": "cifar10",
        # model
        "model_cfg": {
            "architecture": "resnet",
            "n_layers": 3,
        },
        "init_from": "scratch",
        # taining
        "max_iters": 3000,
        "eval_interval": 100,
        "eval_iters": 5,
        "batch_size": 512,
        "optim_cfg": {
            "algo": "sgd",
            "lr": 0.03,
            "weight_decay": 0.0005,
            "momentum": 0.9,
        },
        "grad_accumulation_steps": 1,
    },
}
