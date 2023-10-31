import numpy as np
import itertools
import pandas as pd
import torch


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


def get_new_pairs(X, Y, pmat):
    n = len(X)
    x_bins, y_bins = pmat.shape

    # get probability distribution over pairs of examples
    df1 = pd.DataFrame(
        {
            "ind": list(itertools.product(np.arange(n), np.arange(n))),
            "bins": list(itertools.product(X, Y)),
        }
    )
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
    idx = np.random.choice(np.arange(n**2), size=(n,), replace=True, p=weight)
    return df["ind"].iloc[idx]


def raking_resample(x, y, cx, cy, marginals, num_rounds):
    if is_good_event(cx, cy, marginals):
        pmat = raking_ratio(cx, cy, marginals, num_rounds)
        new_pairs = get_new_pairs(cx, cy, pmat)
        inds = torch.tensor(list(new_pairs))
        return x[inds[:, 0]], y[inds[:, 1]]
    else:
        return x, y
