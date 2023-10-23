configs = [
    {
        # experiment
        "experiment_group": "debug",
        "experiment_name": "debug",
        "id": 0,
        "log_dir": "logs/",
        "output_dir": "/mnt/ssd/ronak/",
        # data
        "dataset": "cifar10",
        "data_dir": "",
        # model
        "model_cfg": {
            "architecture": "myrtle_net",
            "n_layers": 3,
            "residual_blocks": [0, 2],
        },
        # taining
        "max_iter": 10,
        "eval_iterval": 2,
        "batch_size": 512,
        "device": "cuda:0",
        "optim_cfg": {
            "optimizer": "adam",
            "lr": 0.003,
        },
        "grad_accumulation_steps": 1,
        "metrics": ["loss", "accuracy"],
    },
]
