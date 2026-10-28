import numpy as np
from sklearn.tree import DecisionTreeClassifier
from lot.data import parity, rotatedParity, diagonal, orthant, trunk, ringnorm
from lot.tree import LOT, schedules
from lot.metrics import objective, countExtra


def error(m, X, y):
    return np.mean(m.predict(X) != y)


def testGreedyMatchesSklearn():
    """spec 7.8 item 6: an independent entropy tree on the same binary columns; sklearn breaks gain ties
    by a random feature order, ours by lowest index, so one of 20 orders must match"""
    for X, y in [trunk(800, 6, 0), trunk(800, 10, 1), ringnorm(800, 2)]:
        for depth in (2, 3, 4):
            m = LOT(depth=depth, lam=1e-9).fit(X, y)
            errs = set()
            for rs in range(20):
                sk = DecisionTreeClassifier(criterion="entropy", max_depth=depth, min_samples_leaf=5, random_state=rs)
                errs.add(np.mean(sk.fit(m.B, y).predict(m.B) != y))
            assert error(m, X, y) in errs


def testCostEqualsObjective():
    """spec 7.8 item 3: the cost fit() returns is R(T) recomputed from the finished tree"""
    for X, y in [parity(200, 2, 3, 0), rotatedParity(200, 2, 0), trunk(200, 6, 2)]:
        for la in (False, True):
            for ob in (False, True):
                m = LOT(depth=3, gamma=0.003, lookahead=la, oblique=ob).fit(X, y)
                assert abs(m.cost - objective(m, X, y)) < 1e-9


def lookNoWorse(X, y, **kw):
    g = LOT(lookahead=False, **kw).fit(X, y)
    l = LOT(lookahead=True, **kw).fit(X, y)
    assert l.cost <= g.cost + 1e-12, kw


def smallDatasets(seed, n):
    return [parity(n, 2, 3, seed), parity(n, 3, 2, seed), rotatedParity(n, 2, seed),
            diagonal(n, 2, seed), orthant(n, 3, 1, seed), trunk(n, 8, seed)]


def testProp12Axis():
    """R(Look) <= R(Greedy)"""
    for seed in range(5):
        for X, y in smallDatasets(seed, 400):
            for depth in (2, 4):
                lookNoWorse(X, y, depth=depth)


def testProp12Oblique():
    """a failure here points at the per-node seeding of the projections"""
    for seed in range(3):
        for X, y in smallDatasets(seed, 250):
            lookNoWorse(X, y, depth=2, gamma=0.0025, oblique=True, seed=seed)
    X, y = rotatedParity(200, 2, 0)
    for s in schedules.values():
        lookNoWorse(X, y, depth=3, oblique=True, s=s)


def testRotatedParityOnlyLot():
    """spec 6.7: the right root split has zero gain, so only LOT finds the Figure 4 tree"""
    for seed in range(4):
        X, y = rotatedParity(1000, 0, seed)
        Xt, yt = rotatedParity(20000, 0, 100 + seed)
        lot = LOT(depth=2, lookahead=True, oblique=True).fit(X, y)
        assert lot.nLeaves() == 4 and error(lot, Xt, yt) < 0.05
        assert error(LOT(depth=2, lookahead=True).fit(X, y), Xt, yt) > 0.3


def testLargeGammaRemovesObliqueSplits():
    """Look falls back to axis splits; Greedy picks by gain alone (Algorithm 1), so it ends at a leaf"""
    X, y = diagonal(1000, 0, 1)
    look = LOT(depth=3, gamma=1.0, lookahead=True, oblique=True).fit(X, y)
    assert countExtra(look.tree) == 0 and look.nLeaves() > 2
    assert LOT(depth=3, gamma=1.0, oblique=True).fit(X, y).nLeaves() == 1


def testGammaThresholdOnRotatedParity():
    """Figure 4 tree: 3 gamma + 4 lam; oblique root over two axis splits: gamma + 4 lam + 1/4,
    so Figure 4 needs gamma < 1/8, stricter than the leaf bound of spec 6.7"""
    for seed in range(2):
        X, y = rotatedParity(1000, 0, seed)
        assert countExtra(LOT(depth=2, gamma=0.08, lookahead=True, oblique=True).fit(X, y).tree) == 3
        assert countExtra(LOT(depth=2, gamma=0.17, lookahead=True, oblique=True).fit(X, y).tree) <= 1
