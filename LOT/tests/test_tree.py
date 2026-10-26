import numpy as np
from lot.tree import oneHot, entropy, leafCost, gains


def testEntropy():
    assert abs(entropy([5, 5])[0] - 1) < 1e-12
    assert entropy([10, 0])[0] == 0
    assert abs(entropy([1, 1, 1, 1])[0] - 2) < 1e-12


def testLeafCostUsesRootN():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    u, pred = leafCost(oneHot(y, 2), np.array([0, 1, 2, 3]), 0.01)
    assert pred == 0
    """one mistake out of N = 10, not out of 4"""
    assert abs(u - (0.01 + 1 / 10)) < 1e-12


def testGains():
    Y = oneHot(np.array([0, 0, 0, 0, 1, 1, 1, 1]), 2)
    perfect = [1, 1, 1, 1, 0, 0, 0, 0]
    mixed = [1, 1, 1, 0, 1, 0, 0, 0]
    useless = [1, 0, 1, 0, 1, 0, 1, 0]
    g = gains(np.array([perfect, mixed, useless], dtype=bool).T, Y)
    assert abs(g[0] - 1) < 1e-12
    assert abs(g[1] - (1 - 0.8112781244591328)) < 1e-9
    assert abs(g[2]) < 1e-12
