import numpy as np
from lot.data import parity, rotatedParity, orthant, diagonal, trunk
from lot.tree import LOT


def error(m, X, y):
    return np.mean(m.predict(X) != y)


# Look sees past the zero-gain root: on XOR it returns the 4-leaf tree
# (root on one feature, both children on the other)
def testLookSolvesXorAtDepth2():
    for seed in range(8):
        X, y = parity(1000, 2, 0, seed)
        Xt, yt = parity(4000, 2, 0, 100 + seed)
        m = LOT(depth=2, lookahead=True).fit(X, y)
        assert m.nLeaves() == 4
        assert error(m, Xt, yt) < 0.05
        fRoot = m.tree[1][0][0]
        assert m.tree[2][1][0][0] == 1 - fRoot
        assert m.tree[3][1][0][0] == 1 - fRoot


# Proposition 1: R(Look) <= R(Greedy) on the training objective, same candidates
def checkProp1(X, y, depth, lam, gamma):
    g = LOT(depth=depth, lam=lam, gamma=gamma, lookahead=False).fit(X, y)
    l = LOT(depth=depth, lam=lam, gamma=gamma, lookahead=True).fit(X, y)
    assert l.cost <= g.cost + 1e-12


def testProp1ManyDatasets():
    for seed in range(4):
        for lam in (0.001, 0.01):
            for depth in (2, 3, 4):
                X, y = parity(400, 2, 3, seed)
                checkProp1(X, y, depth, lam, 0.0)
                X, y = parity(400, 3, 2, seed)
                checkProp1(X, y, depth, lam, 0.0)
                X, y = orthant(400, 3, 1, seed)
                checkProp1(X, y, depth, lam, 0.0)
                X, y = diagonal(400, 2, seed)
                checkProp1(X, y, depth, lam, 0.0)
                X, y = trunk(400, 6, seed)
                checkProp1(X, y, depth, lam, 0.0)
                X, y = rotatedParity(400, 2, seed)
                checkProp1(X, y, depth, lam, 0.0)