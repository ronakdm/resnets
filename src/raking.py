import numpy as np
import logging
import os


def count_freq(X, Y, marginals):
    # count pairs
    pairs = list(zip(X, Y))
    ind, count = np.unique(pairs, axis=0, return_counts=True)
    cmat = np.zeros((len(marginals[0]), len(marginals[1])), dtype=np.float32)
    cmat[ind[:, 0], ind[:, 1]] = count
    # remove bins with zero mass
    indx, X = np.unique(X, return_inverse=True)
    indy, Y = np.unique(Y, return_inverse=True)
    return X, Y, cmat[np.ix_(indx, indy)], (marginals[0][indx], marginals[1][indy])


def raking_ratio(pmat, marginals, num_iter):
    if np.sum(np.sum(pmat, axis=1) == 0) + np.sum(np.sum(pmat, axis=0) == 0) > 0:
        raise RuntimeError(
            "Missing cluster in the batch; try a smaller quantization level or a larger batch size."
        )
    for _ in range(num_iter):
        pmat = (marginals[0] / np.sum(pmat, axis=1)).reshape(-1, 1) * pmat
        pmat = pmat * (marginals[1] / np.sum(pmat, axis=0))
    return pmat


def get_raking_ratio_weight(idx, quantization, num_rounds):
    marginals = (quantization["x_marginal"], quantization["y_marginal"])
    X = quantization["x_labels"][idx]
    Y = quantization["y_labels"][idx]
    X, Y, cmat, marginals = count_freq(X, Y, marginals)
    # logging.info(f"Batch clusters {len(marginals[0])}, {len(marginals[1])}")
    pmat = raking_ratio(cmat / len(X), marginals, num_rounds)
    prob = pmat[X, Y] / cmat[X, Y]
    return prob / np.sum(prob)