import numpy as np
from lot.data import parity
from lot.binarize import standardize, axisThresholds, binarize


def testStandardize():
    X, y = parity(2000, 2, 1, 0)
    Xs, mu, sd = standardize(X)
    for j in range(3):
        assert abs(Xs[:, j].mean()) < 1e-9
        assert abs(Xs[:, j].std() - 1) < 1e-6
    assert np.allclose(Xs * sd + mu, X)


def testAxisThresholdsCountAndMedian():
    """q = 5 thresholds per feature, and the median is one of them"""
    X, y = parity(2000, 2, 1, 0)
    Xs, mu, sd = standardize(X)
    ax = axisThresholds(Xs, 5)
    assert len(ax) == 3 * 5
    for j in range(3):
        ts = []
        for (jj, t) in ax:
            if jj == j:
                ts.append(t)
        med = np.median(Xs[:, j])
        assert min(abs(t - med) for t in ts) < 0.01


def testEvenQRejected():
    Xs = np.random.default_rng(0).normal(size=(50, 2))
    try:
        axisThresholds(Xs, 4)
        assert False
    except ValueError:
        pass


def testBinarizeMatchesDefinition():
    """columns match the definition, and column f sends about k/(q+1) of the data left"""
    X, y = parity(1200, 2, 0, 0)
    Xs, mu, sd = standardize(X)
    ax = axisThresholds(Xs, 5)
    B = binarize(Xs, ax)
    assert B.shape == (1200, 10)
    assert B.dtype == bool
    for f in range(len(ax)):
        j, t = ax[f]
        for i in range(0, 1200, 97):
            assert B[i, f] == (Xs[i, j] <= t)
    """feature 0, quantiles 1/6 ... 5/6"""
    fr = B[:, :5].mean(axis=0)
    for k in range(5):
        assert abs(fr[k] - (k + 1) / 6) < 0.02


def testBinaryFeatureCollapses():
    """a 0/1 feature has only one useful threshold, so duplicates must collapse"""
    rng = np.random.default_rng(0)
    Xs = np.stack([rng.integers(0, 2, 500).astype(float), rng.normal(size=500)], axis=1)
    ax = axisThresholds(Xs, 5)
    cnt0 = 0
    for (j, t) in ax:
        if j == 0:
            cnt0 += 1
    assert cnt0 == 1