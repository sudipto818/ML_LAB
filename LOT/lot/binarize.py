import numpy as np

def standardize(X):
    """mean 0, std 1, so +-1 coefficients weigh every feature equally; predict reuses mu, sd"""
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-12
    Xs = (X - mu) / sd
    return Xs, mu, sd


def axisThresholds(Xs, q):
    """(j, t) at the quantiles k/(q+1) of every feature"""
    if q % 2 == 0:
        raise ValueError("q must be odd so the median is included")
    p = Xs.shape[1]
    qs = []
    for k in range(1, q + 1):
        qs.append(k / (q + 1))
    ax = []
    for j in range(p):
        """method lower: thresholds are data values"""
        ts = np.unique(np.quantile(Xs[:, j], qs, method="lower"))
        mx = Xs[:, j].max()
        for t in ts:
            """t = max is not a split"""
            if t < mx:
                ax.append((j, t))
    return ax


def binarize(Xs, ax):
    N = len(Xs)
    k = len(ax)
    B = np.zeros((N, k), dtype=bool)
    for f in range(k):
        j, t = ax[f]
        B[:, f] = Xs[:, j] <= t
    return B