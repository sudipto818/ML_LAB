import numpy as np
from lot.data import rotatedParity
from lot.binarize import standardize
from lot.oblique import nProjections, sampleProjections, obliqueCandidates


def testProjectionPool():
    """p = 4: all 12 pairs listed; p = 10: m distinct draws per support size"""
    assert nProjections(4, 2) == 12 and nProjections(6, 3) == 80
    few = sampleProjections(4, 2, 30, np.arange(100), 0, 0)
    assert len(set(few)) == 12 and ((2, 3), (1, -1)) in few
    many = sampleProjections(10, 3, 30, np.arange(100), 0, 0)
    assert len(set(many)) == 60
    assert sum(len(sup) == 2 for sup, sg in many) == 30
    for sup, sg in many:
        assert list(sup) == sorted(set(sup)) and sg[0] == 1


def testCandidateColumns():
    X, y = rotatedParity(600, 2, 0)
    Xs, mu, sd = standardize(X)
    idx = np.arange(0, 600, 2)
    M, meta = obliqueCandidates(Xs, idx, sampleProjections(6, 3, 30, idx, 0, 0), 3)
    assert M.shape == (len(idx), len(meta))
    for f in range(0, len(meta), 7):
        sup, sg, t = meta[f]
        z = Xs[np.ix_(idx, sup)] @ np.array(sg, dtype=float)
        assert np.array_equal(M[:, f], z <= t)
        assert z.min() <= t < z.max()
