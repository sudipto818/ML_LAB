import numpy as np
from lot.data import parity, rotatedParity, diagonal, orthant, trunk
from lot.tree import LOT
from lot.metrics import objective, countExtra


def error(m, X, y):
    return np.mean(m.predict(X) != y)


# s = 1 allows no projection, so the oblique switch must change nothing
def testSOneEqualsAxisOnly():
    X, y = parity(400, 2, 3, 0)
    for la in (False, True):
        a = LOT(depth=3, lookahead=la).fit(X, y)
        b = LOT(depth=3, lookahead=la, oblique=True, s=1).fit(X, y)
        assert a.tree == b.tree and a.cost == b.cost


# y = 1[x1 + x2 > 0]: one oblique split does it (spec Section 5.1), while an axis tree
# with S leaves keeps error >= 1/(2S) (Proposition 9)
def testDiagonalNeedsOneObliqueSplit():
    for seed in range(3):
        X, y = diagonal(1000, 0, seed)
        Xt, yt = diagonal(20000, 0, 100 + seed)
        ob = LOT(depth=4, oblique=True).fit(X, y)
        assert ob.nLeaves() == 2
        assert ob.tree[1][0] == (0, 1) and ob.tree[1][1] == (1, 1)
        assert error(ob, Xt, yt) < 0.02
        ax = LOT(depth=4).fit(X, y)
        assert error(ax, Xt, yt) > 1 / (2 * ax.nLeaves()) - 0.01


# rotated parity at depth 2 (spec Section 6.7): the root split x1 + x2 <= 0 has zero gain, so only
# Look with oblique candidates (LOT) finds the four-leaf tree of Figure 4; axis Look cannot
def testRotatedParityOnlyLot():
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


# Proposition 12 with oblique candidates: R(Look) <= R(Greedy) on the training objective
def testProp12Oblique():
    sets = []
    for seed in range(2):
        sets += [parity(300, 2, 2, seed), rotatedParity(300, 1, seed), diagonal(300, 2, seed),
                 orthant(300, 3, 0, seed), trunk(300, 5, seed)]
    for X, y in sets:
        for depth in (2, 3):
            for gamma in (0.0, 0.0025):
                g = LOT(depth=depth, gamma=gamma, oblique=True).fit(X, y)
                l = LOT(depth=depth, gamma=gamma, lookahead=True, oblique=True).fit(X, y)
                assert l.cost <= g.cost + 1e-12


# the cost the recursion returns equals R(T) recomputed from the finished tree, gamma term included
def testFitCostMatchesObjectiveOblique():
    sets = [rotatedParity(400, 2, 0), diagonal(400, 1, 1), trunk(400, 6, 2)]
    for X, y in sets:
        for la in (False, True):
            m = LOT(depth=3, gamma=0.003, lookahead=la, oblique=True).fit(X, y)
            assert abs(m.cost - objective(m, X, y)) < 1e-9
    assert countExtra(LOT(depth=2, oblique=True).fit(*diagonal(400, 0, 0)).tree) == 1


# same data and seed -> same tree, also when p is large enough that projections are sampled
def testDeterministicOblique():
    X, y = trunk(300, 10, 0)
    a = LOT(depth=3, lookahead=True, oblique=True, seed=5).fit(X, y)
    b = LOT(depth=3, lookahead=True, oblique=True, seed=5).fit(X, y)
    assert a.tree == b.tree


# every oblique split also leaves at least nMin samples on each side
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
