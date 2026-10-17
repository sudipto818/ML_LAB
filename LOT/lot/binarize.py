import numpy as np

def standardize(X):
    """scale every feature to mean 0 and std 1, so one threshold scale fits all features
    also returns mu and sd, because predict has to scale new data the same way
    crucial for oblique splits"""
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-12
    Xs = (X - mu) / sd
    return Xs, mu, sd


def axisThresholds(Xs, q):
    """axis-aligned thresholds at the quantiles 1/(q+1), 2/(q+1), ..., q/(q+1) of each feature
    q must be odd so that the median (the 1/2 quantile) is always a candidate
    returns a list of (feature j, threshold t)"""
    if q % 2 == 0:
        raise ValueError("q must be odd so the median is included")
    p = Xs.shape[1]
    qs = []
    for k in range(1, q + 1):
        qs.append(k / (q + 1))
    ax = []
    for j in range(p):
        """real data values only"""
        ts = np.unique(np.quantile(Xs[:, j], qs, method="lower"))
        mx = Xs[:, j].max()
        for t in ts:
            """t = max would send everything left, not a split"""
            if t < mx:
                ax.append((j, t))
    return ax


def binarize(Xs, ax):
    """B[i, f] = True if sample i goes left under test f, i.e. Xs[i, j] <= t"""
    N = len(Xs)
    k = len(ax)
    B = np.zeros((N, k), dtype=bool)
    for f in range(k):
        j, t = ax[f]
        B[:, f] = Xs[:, j] <= t
    return B