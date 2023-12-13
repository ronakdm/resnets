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
    "fashion_mnist": {},
    "sst2": {},
}


