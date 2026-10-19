import numpy as np
from lot.data import parity, rotatedParity, diagonal, trunk
from lot.tree import LOT
from lot.metrics import objective, countExtra


def error(m, X, y):
    return np.mean(m.predict(X) != y)


def testSOneEqualsAxisOnly():
    X, y = parity(400, 2, 3, 0)
    for la in (False, True):
        a = LOT(depth=3, lookahead=la).fit(X, y)
        b = LOT(depth=3, lookahead=la, oblique=True, s=1).fit(X, y)
        assert a.tree == b.tree and a.cost == b.cost


def testDiagonalNeedsOneObliqueSplit():
    """spec 5.1: one oblique split is exact, an axis tree keeps error >= 1/(2S) (Proposition 9)"""
    for seed in range(3):
        X, y = diagonal(1000, 0, seed)
        Xt, yt = diagonal(20000, 0, 100 + seed)
        ob = LOT(depth=4, oblique=True).fit(X, y)
        assert ob.nLeaves() == 2
        assert ob.tree[1][0] == (0, 1) and ob.tree[1][1] == (1, 1)
        assert error(ob, Xt, yt) < 0.02
        ax = LOT(depth=4).fit(X, y)
        assert error(ax, Xt, yt) > 1 / (2 * ax.nLeaves()) - 0.01


def testRotatedParityOnlyLot():
    """spec 6.7: the right root split has zero gain, so only LOT finds the Figure 4 tree"""
    axErrs = []
    for seed in range(4):
        X, y = rotatedParity(1000, 0, seed)
        Xt, yt = rotatedParity(20000, 0, 100 + seed)
        lot = LOT(depth=2, lookahead=True, oblique=True).fit(X, y)
        assert lot.nLeaves() == 4
        assert error(lot, Xt, yt) < 0.05
        root = lot.tree[1][0]
        other = (2, 3) if root == (0, 1) else (0, 1)
        assert lot.tree[2][1][0] == other and lot.tree[3][1][0] == other
        ax = LOT(depth=2, lookahead=True).fit(X, y)
        axErrs.append(error(ax, Xt, yt))
    assert np.mean(axErrs) > 0.35


def testFitCostMatchesObjectiveOblique():
    sets = [rotatedParity(400, 2, 0), diagonal(400, 1, 1), trunk(400, 6, 2)]
    for X, y in sets:
        for la in (False, True):
            m = LOT(depth=3, gamma=0.003, lookahead=la, oblique=True).fit(X, y)
            assert abs(m.cost - objective(m, X, y)) < 1e-9
    assert countExtra(LOT(depth=2, oblique=True).fit(*diagonal(400, 0, 0)).tree) == 1


def testNMinRespectedOblique():
    X, y = rotatedParity(300, 2, 0)
    m = LOT(depth=4, oblique=True, nMin=40).fit(X, y)
    def walk(node, idx):
        if node[0] == "leaf":
            assert len(idx) >= 40
            return
        sup, sg, t = node[1]
        left = m.Xs[np.ix_(idx, sup)] @ np.array(sg, dtype=float) <= t
        walk(node[2], idx[left])
        walk(node[3], idx[~left])
    walk(m.tree, np.arange(len(y)))
