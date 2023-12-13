defaults = {
    "cifar10":  {
        "data": {
            "data_dir": "/mnt/ssd/ronak/datasets/",
            "unbalance": 1.0,
            "quantization_x": "convnext_base_kmeans_50/image_labels.npy",
            "quantization_y": "convnext_base_kmeans_50/class_labels.npy",
        },
        "model": {
            "architecture": "resnet",
            "n_layers": 2,
            "n_classes": 10,
            "height": 32,
            "width": 32,
        },
        "optim": {
            "algo": "sgd",
            "lr": 1e-2,
            "momentum": 0.0,
            "weight_decay": 0.0,
        },
        "training": {
            "log_dir": "logs/",
            "output_dir": "/mnt/ssd/ronak/output",
            "init_from": "scratch",
            "max_iters": 6000,
            "eval_interval": 400,
            "eval_iters": 400,
            "batch_size": 128,
            "grad_accumulation_steps": 1,
            "track_variance": True,
            "resample": False,
        }
    },
    "fashion_mnist": {
        "data": {
            "data_dir": "/mnt/ssd/ronak/datasets/",
            "unbalance": 1.0,
            "quantization_x": "convnext_base_kmeans_50/image_labels.npy",
            "quantization_y": "convnext_base_kmeans_50/class_labels.npy",
        },
        "model": {
            "architecture": "resnet",
            "in_channels": 1,
            "n_layers": 2,
            "n_classes": 10,
            "height": 28,
            "width": 28,
        },
        "optim": {
            "algo": "sgd",
            "lr": 1e-2,
            "momentum": 0.0,
            "weight_decay": 0.0,
        },
        "training": {
            "log_dir": "logs/",
            "output_dir": "/mnt/ssd/ronak/output",
            "init_from": "scratch",
            "max_iters": 4000,
            "eval_interval": 400,
            "eval_iters": 200,
            "batch_size": 128,
            "grad_accumulation_steps": 1,
            "track_variance": True,
            "resample": False,
        }
    },
    "sst2": {
        "data": {
            "data_dir": "/mnt/ssd/ronak/datasets/",
            "unbalance": 1.0,
        },
        "model": {
            "architecture": "transformer",
            "block_size": 77,
            "vocab_size": 49408,
            "n_layer": 2,
            "n_head": 8,
            "n_embd": 64,
            "n_classes": 2,
            "dropout": 0.0,
            "bias": False,
        },
        "optim": {
            "algo": "sgd",
            "lr": 1e-3,
            "momentum": 0.0,
            "weight_decay": 0.0,
        },
        "training": {
            "log_dir": "logs/",
            "output_dir": "/mnt/ssd/ronak/output",
            "init_from": "scratch",
            "max_iters": 4000,
            "eval_interval": 200,
            "eval_iters": 100,
            "batch_size": 128,
            "grad_accumulation_steps": 1,
            "track_variance": True,
            "resample": False,
        }
    },
}