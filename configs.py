configs = [
    {
        # experiment
        "experiment_group": "debug",
        "experiment_name": "debug",
        "seed": 0,
        "log_dir": "logs/",
        "output_dir": "/mnt/ssd/ronak/output",
        # data
        "dataset": "cifar10",
        # model
        "model_cfg": {
            "architecture": "myrtle_net",
            "n_layers": 3,
            "residual_blocks": [0, 2],
        },
        # taining
        "max_iters": 30,
        "eval_interval": 10,
        "eval_iters": 5,
        "batch_size": 512,
        "device": "cuda:0",
        "optim_cfg": {
            "optimizer": "adam",
            "lr": 0.003,
        },
        "grad_accumulation_steps": 1,
    },
]
