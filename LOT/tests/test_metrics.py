import numpy as np
from lot.data import parity, rotatedParity, orthant
from lot.tree import LOT
from lot.metrics import (errorRate, countLeaves, countSplits, countNnz,countExtra, objective)


def testErrorRate():
    assert errorRate([0, 1, 1, 0], [0, 1, 0, 0]) == 0.25
    assert errorRate([1, 1], [1, 1]) == 0.0


# a tree built by hand:  root test x1 + x2 <= 5.5 (a pair, so c = 1), then
# the left leaf predicts 0 and the right branch is split again by x1 <= 8.5 (axis, c = 0)
def handModel():
    m = LOT(depth=2, lam=0.01, gamma=0.1)
    m.mu = np.zeros(2)
    m.sd = np.ones(2)
    right = ("split", ((0,), (1,), 8.5), ("leaf", 1), ("leaf", 0))
    m.tree = ("split", ((0, 1), (1, 1), 5.5), ("leaf", 0), right)
    return m


def testCountsOnHandTree():
    m = handModel()
    assert countLeaves(m.tree) == 3
    assert countSplits(m.tree) == 2
    assert countNnz(m.tree) == 3              # 2 (the pair) + 1 (the axis test)
    assert countExtra(m.tree) == 1            # only the pair costs extra


# Manual computation on 10 points, x1 = 1..10 and x2 = 0, labels below:
#   x1:  1 2 3 4 5 | 6 7 8 | 9 10
#   y :  0 0 0 0 1 | 1 1 0 | 0 0
#   left leaf (x1 <= 5) predicts 0 -> 1 mistake (x1 = 5)
#   middle leaf (6..8) predicts 1  -> 1 mistake (x1 = 8)
#   right leaf (9, 10) predicts 0  -> 0 mistakes
#   error = 2/10 = 0.2;   R = 0.2 + 0.01 * 3 + 0.1 * 1 = 0.33
def testObjectiveByHand():
    X = np.zeros((10, 2))
    X[:, 0] = np.arange(1, 11)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0])
    m = handModel()
    assert errorRate(y, m.predict(X)) == 0.2
    assert abs(objective(m, X, y) - 0.33) < 1e-12


# the cost that fit() accumulates must equal R(T) recomputed from the finished tree
def testFitCostMatchesObjective():
    sets = [parity(500, 2, 3, 0), parity(500, 3, 1, 1), orthant(500, 3, 1, 2), rotatedParity(500, 1, 3)]
    for X, y in sets:
        for la in (False, True):
            for gamma in (0.0, 0.003):
                m = LOT(depth=3, lam=0.005, gamma=gamma, lookahead=la).fit(X, y)
                assert abs(m.cost - objective(m, X, y)) < 1e-9
                assert countLeaves(m.tree) == m.nLeaves()