import numpy as np
from lot.data import parity
from lot.tree import oneHot, entropy, leafCost, gains


def testEntropyHandValues():
    assert abs(entropy([5, 5])[0] - 1.0) < 1e-12
    assert abs(entropy([10, 0])[0] - 0.0) < 1e-12
    assert abs(entropy([1, 1, 1, 1])[0] - 2.0) < 1e-12
    """empty node"""
    assert entropy([0, 0])[0] == 0.0
    h = entropy(np.array([[5, 5], [10, 0], [3, 1]]))
    assert abs(h[2] - 0.8112781244591328) < 1e-9          


def testLeafCost():
    """N = 10 samples in total, the node holds 4 of them with counts [3, 1]"""
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    Y = oneHot(y, 2)
    idx = np.array([0, 1, 2, 3])
    u, pred = leafCost(Y, idx, 0.01)
    assert pred == 0
    """divided by root N, not by 4"""
    assert abs(u - (0.01 + 1 / 10)) < 1e-12
    """pure node costs only lam"""
    u, pred = leafCost(Y, np.array([0, 1, 2]), 0.01)
    assert abs(u - 0.01) < 1e-12


def testGainHandValues():
    """8 samples, 4 of each class"""
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    Yd = oneHot(y, 2)
    perfect = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
    """left (3,1), right (1,3)"""
    mixed = np.array([1, 1, 1, 0, 1, 0, 0, 0], dtype=bool)
    """left (2,2), right (2,2)"""
    useless = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=bool)
    M = np.stack([perfect, mixed, useless], axis=1)
    g = gains(M, Yd)
    assert abs(g[0] - 1.0) < 1e-12
    assert abs(g[1] - (1 - 0.8112781244591328)) < 1e-9
    assert abs(g[2] - 0.0) < 1e-12


def testXorZeroGainThenFullGain():
    """on XOR no single-feature test has gain, but after one split the second feature does"""
    X, y = parity(4000, 2, 0, 0)
    Y = oneHot(y, 2)
    qs = [0.25, 0.5, 0.75]
    cols = []
    for j in range(2):
        for q in qs:
            t = np.quantile(X[:, j], q)
            cols.append(X[:, j] <= t)
    M = np.stack(cols, axis=1)
    g = gains(M, Y)
    for f in range(len(g)):
        assert g[f] < 0.01
    left = np.where(X[:, 0] <= 0)[0]                    
    M2 = (X[left, 1] <= 0).reshape(-1, 1)
    g2 = gains(M2, Y[left])
    assert g2[0] > 0.95