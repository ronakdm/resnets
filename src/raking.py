import torch


def is_good_event(cx, cy, px, py):
    return len(torch.unique(cx)) == len(px) and len(torch.unique(cy)) == len(py)


def raking_ratio(cx, cy, px, py, num_iter):
    pmat = cx[:, None]
    if is_good_event(cx, cy, px, py):
        pmat = count_freq(X, Y, (len(marginals[0]), len(marginals[1])))
        if np.sum(np.sum(pmat, axis=1) == 0) + np.sum(np.sum(pmat, axis=0) == 0) > 0:
            raise RuntimeError("Missing mass in this sample. Try a larger sample size.")

        est = [pmat]
        for _ in range(num_iter):
            pmat = (marginals[0] / np.sum(pmat, axis=1)).reshape(-1, 1) * pmat
            pmat = pmat * (marginals[1] / np.sum(pmat, axis=0))
            est.append(pmat)
    return est
