import numpy as np
import itertools
import pandas as pd
import torch
import math


def is_good_event(cx, cy, marginals):
    return len(np.unique(cx)) == len(marginals[0]) and len(np.unique(cy)) == len(
        marginals[1]
    )


def count_freq(X, Y, sizes):
    # count pairs
    pairs = list(zip(X, Y))
    ind, count = np.unique(pairs, axis=0, return_counts=True)
    cmat = np.zeros(sizes)
    cmat[ind[:, 0], ind[:, 1]] = count
    return cmat / len(pairs)


def raking_ratio(X, Y, marginals, num_rounds):
    pmat = count_freq(X, Y, (len(marginals[0]), len(marginals[1])))
    est = [pmat]
    for _ in range(num_rounds):
        pmat = (marginals[0] / np.sum(pmat, axis=1)).reshape(-1, 1) * pmat
        pmat = pmat * (marginals[1] / np.sum(pmat, axis=0))
        est.append(pmat)
    return est[-1]


def get_new_indices(X, Y, pmat):
    n = len(X)
    x_bins, y_bins = pmat.shape

    # get probability distribution over pairs of examples
    df1 = pd.DataFrame({"ind": np.arange(n), "bins": [(x, y) for x, y in zip(X, Y)]})
    df2 = pd.DataFrame(
        {
            "bins": list(itertools.product(np.arange(x_bins), np.arange(y_bins))),
            "prob": pmat.reshape(-1),
        }
    )
    bin_names, bin_counts = np.unique(df1["bins"], return_counts=True)
    df = df1.merge(df2, on="bins", how="left").merge(
        pd.DataFrame({"bins": bin_names, "counts": bin_counts}), on="bins", how="left"
    )
    weight = df["prob"].to_numpy() / np.maximum(1.0, df["counts"].to_numpy())

    # sample indices
    idx = np.random.choice(np.arange(n), size=(n,), replace=True, p=weight)
    return torch.tensor(idx)


def get_new_weights(X, Y, pmat):
    n = len(X)
    x_bins, y_bins = pmat.shape

    # get probability distribution over pairs of examples
    df1 = pd.DataFrame({"ind": np.arange(n), "bins": [(x, y) for x, y in zip(X, Y)]})
    df2 = pd.DataFrame(
        {
            "bins": list(itertools.product(np.arange(x_bins), np.arange(y_bins))),
            "prob": pmat.reshape(-1),
        }
    )
    bin_names, bin_counts = np.unique(df1["bins"], return_counts=True)
    df = df1.merge(df2, on="bins", how="left").merge(
        pd.DataFrame({"bins": bin_names, "counts": bin_counts}), on="bins", how="left"
    )
    weight = df["prob"].to_numpy() / np.maximum(1.0, df["counts"].to_numpy())
    assert np.abs(weight.sum() - 1.0) < 1e-8
    return torch.tensor(weight).float()


def get_raking_weights(x, y, cx, cy, marginals, num_rounds):
    if is_good_event(cx, cy, marginals):
        pmat = raking_ratio(cx, cy, marginals, num_rounds)
        return get_new_weights(cx, cy, pmat)
    else:
        return torch.ones(len(x)) / len(x)
