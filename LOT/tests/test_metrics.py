import numpy as np
from lot.tree import LOT
from lot.metrics import objective, kappa


def testObjectiveByHand():
    """root x1 + x2 <= 5.5 (c = 1), right child x1 <= 8.5; leaves predict 0 | 1 | 0, wrong at x1 = 5 and 8
    R = 2/10 + 0.01 * 3 + 0.1 * 1 = 0.33"""
    m = LOT(depth=2, lam=0.01, gamma=0.1)
    m.mu = np.zeros(2)
    m.sd = np.ones(2)
    right = ("split", ((0,), (1,), 8.5), ("leaf", 1), ("leaf", 0))
    m.tree = ("split", ((0, 1), (1, 1), 5.5), ("leaf", 0), right)
    X = np.zeros((10, 2))
    X[:, 0] = np.arange(1, 11)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0])
    assert abs(objective(m, X, y) - 0.33) < 1e-12


def testKappa():
    """po = 3/4, pe = 1/2 * 1/4 + 1/2 * 3/4 = 1/2"""
    assert abs(kappa([0, 0, 1, 1], [0, 1, 1, 1]) - 0.5) < 1e-12
    assert kappa([0, 0, 1, 1], [1, 1, 1, 1]) == 0
