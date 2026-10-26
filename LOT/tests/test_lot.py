import numpy as np
from lot.data import parity, rotatedParity, diagonal, trunk
from lot.oblique import project
from lot.tree import LOT
from lot.metrics import objective


def error(m, X, y):
    return np.mean(m.predict(X) != y)


def leafSizes(m, node, idx, out):
    if node[0] == "leaf":
        out.append(len(idx))
        return out
    sup, sg, t = node[1]
    left = project(m.Xs, idx, sup, sg) <= t
    leafSizes(m, node[2], idx[left], out)
    leafSizes(m, node[3], idx[~left], out)
    return out


def testLookSolvesXor():
    """spec Example 7: the root has zero gain, Look still finds the four-leaf tree"""
    for seed in range(4):
        X, y = parity(1000, 2, 0, seed)
        m = LOT(depth=2, lookahead=True).fit(X, y)
        assert m.nLeaves() == 4
        assert error(m, *parity(4000, 2, 0, 100 + seed)) < 0.05


def testDiagonalNeedsOneObliqueSplit():
    """spec 5.1: one oblique split is exact, an axis tree keeps error >= 1/(2S) (Proposition 9)"""
    for seed in range(3):
        X, y = diagonal(1000, 0, seed)
        Xt, yt = diagonal(20000, 0, 100 + seed)
        ob = LOT(depth=4, oblique=True).fit(X, y)
        assert ob.nLeaves() == 2 and error(ob, Xt, yt) < 0.02
        ax = LOT(depth=4).fit(X, y)
        assert error(ax, Xt, yt) > 1 / (2 * ax.nLeaves()) - 0.01


def testRotatedParityOnlyLot():
    """spec 6.7: the right root split has zero gain, so only LOT finds the Figure 4 tree"""
    for seed in range(4):
        X, y = rotatedParity(1000, 0, seed)
        Xt, yt = rotatedParity(20000, 0, 100 + seed)
        lot = LOT(depth=2, lookahead=True, oblique=True).fit(X, y)
        assert lot.nLeaves() == 4 and error(lot, Xt, yt) < 0.05
        assert error(LOT(depth=2, lookahead=True).fit(X, y), Xt, yt) > 0.3


def testCostEqualsObjective():
    """spec 7.8 item 3: the cost the recursion returns is R(T) recomputed from the finished tree"""
    for X, y in [parity(200, 2, 3, 0), rotatedParity(200, 2, 0), trunk(200, 6, 2)]:
        for la in (False, True):
            for ob in (False, True):
                m = LOT(depth=3, gamma=0.003, lookahead=la, oblique=ob).fit(X, y)
                assert abs(m.cost - objective(m, X, y)) < 1e-9


def testSameSeedSameTree():
    X, y = trunk(300, 10, 0)
    for la in (False, True):
        a = LOT(depth=2, lookahead=la, oblique=True, seed=5).fit(X, y)
        b = LOT(depth=2, lookahead=la, oblique=True, seed=5).fit(X, y)
        assert a.tree == b.tree


def testNMinRespected():
    X, y = rotatedParity(300, 2, 0)
    m = LOT(depth=4, oblique=True, nMin=40).fit(X, y)
    assert min(leafSizes(m, m.tree, np.arange(300), [])) >= 40
