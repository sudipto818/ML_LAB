import numpy as np
from lot.binarize import axisThresholds


def testAxisThresholds():
    """q = 5 per feature with the median among them; a 0/1 feature collapses to one threshold"""
    rng = np.random.default_rng(0)
    Xs = np.stack([rng.normal(size=500), rng.integers(0, 2, 500).astype(float)], axis=1)
    ax = axisThresholds(Xs, 5)
    t0 = [t for j, t in ax if j == 0]
    assert len(t0) == 5
    assert min(abs(t - np.median(Xs[:, 0])) for t in t0) < 0.01
    assert len([t for j, t in ax if j == 1]) == 1
