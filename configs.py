configs = [
    {
        "model_id": "01",
        "n_layers": 1,
        "residual_blocks": [],
        "n_epochs": 24,
        "device": "cuda:0",
        "model_name": "myrtle_net",
        "experiment_name": "loss_curve",
        "logs_dir": "logs/",
        "output_dir": "/mnt/hdd/ronak/cifar10_resnets",
        "batch_size": 512,
        "lr": 0.003,
        "metrics": [
            "loss",
            "accuracy"
        ]
    },
    {
        "model_id": "02",
        "n_layers": 1,
        "residual_blocks": [
            0
        ],
        "n_epochs": 24,
        "device": "cuda:1",
        "model_name": "myrtle_net",
        "experiment_name": "loss_curve",
        "logs_dir": "logs/",
        "output_dir": "/mnt/hdd/ronak/cifar10_resnets",
        "batch_size": 512,
        "lr": 0.003,
        "metrics": [
            "loss",
            "accuracy"
        ]
    },
    {
        "model_id": "03",
        "n_layers": 2,
        "residual_blocks": [],
        "n_epochs": 24,
        "device": "cuda:2",
        "model_name": "myrtle_net",
        "experiment_name": "loss_curve",
        "logs_dir": "logs/",
        "output_dir": "/mnt/hdd/ronak/cifar10_resnets",
        "batch_size": 512,
        "lr": 0.003,
        "metrics": [
            "loss",
            "accuracy"
        ]
    },
    {
        "model_id": "04",
        "n_layers": 2,
        "residual_blocks": [
            0
        ],
        "n_epochs": 24,
        "device": "cuda:3",
        "model_name": "myrtle_net",
        "experiment_name": "loss_curve",
        "logs_dir": "logs/",
        "output_dir": "/mnt/hdd/ronak/cifar10_resnets",
        "batch_size": 512,
        "lr": 0.003,
        "metrics": [
            "loss",
            "accuracy"
        ]
    },
    {
        "model_id": "05",
        "n_layers": 2,
        "residual_blocks": [
            1
        ],
        "n_epochs": 24,
        "device": "cuda:0",
        "model_name": "myrtle_net",
        "experiment_name": "loss_curve",
        "logs_dir": "logs/",
        "output_dir": "/mnt/hdd/ronak/cifar10_resnets",
        "batch_size": 512,
        "lr": 0.003,
        "metrics": [
            "loss",
            "accuracy"
        ]
    },
    {
        "model_id": "06",
        "n_layers": 2,
        "residual_blocks": [
            0,
            1
        ],
        "n_epochs": 24,
        "device": "cuda:1",
        "model_name": "myrtle_net",
        "experiment_name": "loss_curve",
        "logs_dir": "logs/",
        "output_dir": "/mnt/hdd/ronak/cifar10_resnets",
        "batch_size": 512,
        "lr": 0.003,
        "metrics": [
            "loss",
            "accuracy"
        ]
    },
    {
        "model_id": "07",
        "n_layers": 3,
        "residual_blocks": [],
        "n_epochs": 24,
        "device": "cuda:2",
        "model_name": "myrtle_net",
        "experiment_name": "loss_curve",
        "logs_dir": "logs/",
        "output_dir": "/mnt/hdd/ronak/cifar10_resnets",
        "batch_size": 512,
        "lr": 0.003,
        "metrics": [
            "loss",
            "accuracy"
        ]
    },
    {
        "model_id": "08",
        "n_layers": 3,
        "residual_blocks": [
            0
        ],
        "n_epochs": 24,
        "device": "cuda:3",
        "model_name": "myrtle_net",
        "experiment_name": "loss_curve",
        "logs_dir": "logs/",
        "output_dir": "/mnt/hdd/ronak/cifar10_resnets",
        "batch_size": 512,
        "lr": 0.003,
        "metrics": [
            "loss",
            "accuracy"
        ]
    }
]