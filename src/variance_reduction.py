import torch
import numpy as np
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


def get_raking_ratio_weight(idx, quantization, num_rounds, resample=False):
    if resample:
        nlabels =  len(quantization["y_marginal"])
        marginals = (quantization["x_marginal"], np.ones(nlabels) / nlabels)
    else:
        marginals = (quantization["x_marginal"], quantization["y_marginal"])
    X = quantization["x_labels"][idx]
    Y = quantization["y_labels"][idx]
    X, Y, cmat, marginals = count_freq(X, Y, marginals)
    pmat = raking_ratio(cmat / len(X), marginals, num_rounds)
    prob = pmat[X, Y] / cmat[X, Y]
    return prob / np.sum(prob)

def compute_loss(model, idx, X, Y, vr=None):
    if not ("type" in vr):
        loss, _ = model(X, Y)
    elif vr["type"] == "raking":
        quantization = vr["quantization"]
        device = X.get_device()
        sample_weight = torch.from_numpy(get_raking_ratio_weight(
            idx, quantization, vr["num_rounds"], resample=vr['resample']
        )).float().to(device=X.get_device(), non_blocking=True)
        loss, _ = model(X, Y, sample_weight=sample_weight.to(device))
    return loss

def compute_gradients(parameters, loss, vr=None, quantization=None):
    return torch.autograd.grad(outputs=loss, inputs=parameters)