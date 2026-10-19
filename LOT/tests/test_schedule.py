import numpy as np
from lot.data import rotatedParity, diagonal
from lot.tree import LOT, schedules
from lot.metrics import countExtra


def error(m, X, y):
    return np.mean(m.predict(X) != y)


def splitsByLevel(node, level=0, out=None):
    """(level, ||a||_0) of every split"""
    if out is None:
        out = []
    if node[0] == "split":
        out.append((level, len(node[1][0])))
        splitsByLevel(node[2], level + 1, out)
        splitsByLevel(node[3], level + 1, out)
    return out


def testIntSEqualsFlatSchedule():
    X, y = rotatedParity(300, 2, 0)
    a = LOT(depth=3, lookahead=True, oblique=True, s=3).fit(X, y)
    b = LOT(depth=3, lookahead=True, oblique=True, s=(3, 3, 3)).fit(X, y)
    assert a.tree == b.tree and a.cost == b.cost


def testScheduleBoundsEveryLevel():
    X, y = rotatedParity(600, 2, 0)
    for name, s in schedules.items():
        m = LOT(depth=4, oblique=True, s=s).fit(X, y)
        lv = splitsByLevel(m.tree)
        assert len(lv) > 0
        for level, nnz in lv:
            assert nnz <= s[level], name
    m = LOT(depth=3, lookahead=True, oblique=True, s=(1, 1, 3)).fit(X, y)
    for level, nnz in splitsByLevel(m.tree):
        assert nnz <= (1, 1, 3)[level]


def testTailHeavyMissesRotatedParity():
    """the oblique split must be at the root (H5), so s = (1, 3) fails and (2, 2) succeeds"""
    for seed in range(2):
        X, y = rotatedParity(1000, 0, seed)
        Xt, yt = rotatedParity(20000, 0, 100 + seed)
        tail = LOT(depth=2, lookahead=True, oblique=True, s=(1, 3)).fit(X, y)
        assert len(tail.tree[1][0]) == 1
        assert error(tail, Xt, yt) > 0.2
        pairs = LOT(depth=2, lookahead=True, oblique=True, s=(2, 2)).fit(X, y)
        assert pairs.nLeaves() == 4 and countExtra(pairs.tree) == 3
        assert error(pairs, Xt, yt) < 0.05


def testLargeGammaRemovesObliqueSplits():
    """Look falls back to axis splits; Greedy picks by gain alone (Algorithm 1), so it ends at a leaf"""
    X, y = diagonal(1000, 0, 1)
    look = LOT(depth=3, gamma=1.0, lookahead=True, oblique=True).fit(X, y)
    assert countExtra(look.tree) == 0 and look.nLeaves() > 2
    greedy = LOT(depth=3, gamma=1.0, oblique=True).fit(X, y)
    assert countExtra(greedy.tree) == 0 and greedy.nLeaves() == 1


def testGammaThresholdOnRotatedParity():
    """Figure 4 tree: 3 gamma + 4 lam; oblique root over two axis splits: gamma + 4 lam + 1/4,
    so Figure 4 needs gamma < 1/8, stricter than the leaf bound of spec 6.7"""
    for seed in range(2):
        X, y = rotatedParity(1000, 0, seed)
        lo = LOT(depth=2, gamma=0.08, lookahead=True, oblique=True).fit(X, y)
        assert countExtra(lo.tree) == 3
        hi = LOT(depth=2, gamma=0.17, lookahead=True, oblique=True).fit(X, y)
        assert countExtra(hi.tree) <= 1
