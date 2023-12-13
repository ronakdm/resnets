configs = {
    "cifar10": {
        "debug": {
            "training": {
                "max_iters": 300,
                "eval_interval": 100,
                "eval_iters": 5,
                "batch_size": 512,
            },
        },
        "default": {},
        "raking_r2_k50_b128": {
            "variance_reduction": {
                "type": "raking",
                "num_rounds": 2,
            },
        },
    },
    "fashion_mnist": {},
    "sst2": {},
}


