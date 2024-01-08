configs = {
    "cifar10": {
        "debug": {
            "data": {
                "unbalance": 10,
            },
            "training": {
                "max_iters": 300,
                "eval_interval": 100,
                "eval_iters": 5,
                "resample": True,
            },
            "variance_reduction": {
                "type": "raking",
                "num_rounds": 2,
            },
        },
        "default": {},
        "default_b128_u10": {
            "data": {
                "unbalance": 10,
            },
            "training": {
                "resample": True,
            },
        },
        "raking_r2_k50_b128_u10": {
            "data": {
                "unbalance": 10,
            },
            "variance_reduction": {
                "type": "raking",
                "num_rounds": 2,
            },
            "training": {
                "resample": True,
            },
        },
        "default_b64_u10": {
            "data": {
                "unbalance": 10,
            },
            "training": {
                "resample": True,
                "batch_size": 64
            },
        },
        "raking_r2_k50_b64_u10": {
            "data": {
                "unbalance": 10,
            },
            "variance_reduction": {
                "type": "raking",
                "num_rounds": 2,
            },
            "training": {
                "resample": True,
                "batch_size": 64
            },
        },
    },
    "fashion_mnist": {
        "debug": {
            "data": {
                "unbalance": 10,
            },
            "training": {
                "max_iters": 300,
                "eval_interval": 100,
                "eval_iters": 5,
                "resample": True,
                "track_variance": False,
            },
            "variance_reduction": {
                "type": "raking",
                "num_rounds": 2,
            },
        },
        "default": {
            "training": {
                "track_variance": False,
            },
        },
        "default_b128_u10": {
            "data": {
                "unbalance": 10,
            },
            "training": {
                "resample": True,
            },
        },
        "raking_r2_k50_b128_u10": {
            "data": {
                "unbalance": 10,
            },
            "variance_reduction": {
                "type": "raking",
                "num_rounds": 2,
            },
            "training": {
                "resample": True,
            },
        },
        "default_b64_u10": {
            "data": {
                "unbalance": 10,
            },
            "training": {
                "resample": True,
                "batch_size": 64
            },
        },
        "raking_r2_k50_b64_u10": {
            "data": {
                "unbalance": 10,
            },
            "variance_reduction": {
                "type": "raking",
                "num_rounds": 2,
            },
            "training": {
                "resample": True,
                "batch_size": 64
            },
        },
    },
    "ub_fmnist": {
        "debug": {},
        # LL settings (default)
        "default_b256": {
            "training": {
                "track_variance": False,
            }
        },
        # LL settings (raking)
        "raking_r1_k100_b256": {
            "variance_reduction": {
                "type": "raking",
                "num_rounds": 1,
                "use_y_marginal": False,
            },
            "training": {
                "track_variance": False,
            }
        },
        # 2 raking iterations
        "raking_r2_k100_b256": {
            "variance_reduction": {
                "type": "raking",
                "num_rounds": 2,
                "use_y_marginal": False,
            },
            "training": {
                "track_variance": False,
            }
        },
        # 3 raking iterations
        "raking_r3_k100_b256": {
            "variance_reduction": {
                "type": "raking",
                "num_rounds": 3,
                "use_y_marginal": False,
            },
            "training": {
                "track_variance": False,
            }
        },
        # resnet (default)
        "default_b256_resnet": {
            "training": {
                "track_variance": False,
            },
            "model": {
                "architecture": "resnet",
                "in_channels": 1,
                "n_layers": 2,
                "n_classes": 10,
                "height": 28,
                "width": 28,
            },
        },
        # resnet (raking)
        "raking_r1_k100_b256_resnet": {
            "variance_reduction": {
                "type": "raking",
                "num_rounds": 1,
                "use_y_marginal": False,
            },
            "training": {
                "track_variance": False,
            },
            "model": {
                "architecture": "resnet",
                "in_channels": 1,
                "n_layers": 2,
                "n_classes": 10,
                "height": 28,
                "width": 28,
            },
        },
        # cosine decay (default)
        "default_b256_cosine": {
            "training": {
                "track_variance": False,
            },
            "optim": {
                "cosine_decay": True,
            },
        },
        # cosine decay (raking)
        "raking_r1_k100_b256_cosine": {
            "variance_reduction": {
                "type": "raking",
                "num_rounds": 1,
                "use_y_marginal": False,
            },
            "training": {
                "track_variance": False,
            },
            "optim": {
                "cosine_decay": True,
            },
        },
        # momentum (default)
        "default_b256_momentum": {
            "training": {
                "track_variance": False,
            },
            "optim": {
                "momentum": 0.9,
            },
        },
        # momentum (raking)
        "raking_r1_k100_b256_momentum": {
            "variance_reduction": {
                "type": "raking",
                "num_rounds": 1,
                "use_y_marginal": False,
            },
            "training": {
                "track_variance": False,
            },
            "optim": {
                "momentum": 0.9,
            },
        },
        # weight decay (default)
        "default_b256_wd": {
            "training": {
                "track_variance": False,
            },
            "optim": {
                "weight_decay": 0.001,
            },
        },
        # weight decay (raking)
        "raking_r1_k100_b256_wd": {
            "variance_reduction": {
                "type": "raking",
                "num_rounds": 1,
                "use_y_marginal": False,
            },
            "training": {
                "track_variance": False,
            },
            "optim": {
                "weight_decay": 0.001,
            },
        },
        # class marginal
        "raking_r1_k100_b256_ymarg": {
            "variance_reduction": {
                "type": "raking",
                "num_rounds": 1,
                "use_y_marginal": True,
            },
            "training": {
                "track_variance": False,
            },
        },
        # train data used for quantization
        "raking_r1_k100_b256_train_quant": {
            "data": {
                "data_dir": "/mnt/ssd/ronak/datasets/",
                "unbalance": 1.0,
                "x_labels": "convmnist_e24_train_kmeans_100/image_labels.npy",
                "y_labels": "convmnist_e24_train_kmeans_100/class_labels.npy",
                "x_marginal": "convmnist_e24_train_kmeans_100/image_marginal.npy",
                "y_marginal": "convmnist_e24_train_kmeans_100/class_marginal.npy",
            },
            "variance_reduction": {
                "type": "raking",
                "num_rounds": 1,
                "use_y_marginal": False,
            },
            "training": {
                "track_variance": False,
            },
        },
        # use pretrained model on train data
        "raking_r1_k100_b256_convnext_base": {
            "data": {
                "data_dir": "/mnt/ssd/ronak/datasets/",
                "unbalance": 1.0,
                "x_labels": "convnext_base_kmeans_100/image_labels.npy",
                "y_labels": "convnext_base_kmeans_100/class_labels.npy",
                "x_marginal": "convnext_base_kmeans_100/image_marginal.npy",
                "y_marginal": "convnext_base_kmeans_100/class_marginal.npy",
            },
            "variance_reduction": {
                "type": "raking",
                "num_rounds": 1,
                "use_y_marginal": False,
            },
            "training": {
                "track_variance": False,
            },
        },
    },
    "sst2": {},
}


