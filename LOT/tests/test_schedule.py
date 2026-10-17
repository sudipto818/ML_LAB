import numpy as np
from lot.data import rotatedParity, diagonal
from lot.tree import LOT, schedules
from lot.metrics import countExtra


def error(m, X, y):
    return np.mean(m.predict(X) != y)


def splitsByLevel(node, level=0, out=None):
    """(level, number of non-zeros) of every split node"""
    if out is None:
        out = []
    if node[0] == "split":
        out.append((level, len(node[1][0])))
        splitsByLevel(node[2], level + 1, out)
        splitsByLevel(node[3], level + 1, out)
    return out


def testIntSEqualsFlatSchedule():
    """s = 3 is shorthand for the flat schedule (3, 3, 3)"""
    X, y = rotatedParity(300, 2, 0)
    a = LOT(depth=3, lookahead=True, oblique=True, s=3).fit(X, y)
    b = LOT(depth=3, lookahead=True, oblique=True, s=(3, 3, 3)).fit(X, y)
    assert a.tree == b.tree and a.cost == b.cost


def testScheduleBoundsEveryLevel():
    """a split at level l never uses more than s(l) features, for the three schedules of spec 6.8"""
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
    """rotated parity needs its oblique split at the root (H5): with s = (1, 3) the root can only be
    an axis test and LOT fails; with pairs at both levels it finds the four-leaf tree"""
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
    """gamma = 1 makes any oblique split cost more than the whole error, so none survives
    Look falls back to an axis staircase; Greedy picks splits by gain alone (Algorithm 1, line 3),
    so when its best-gain split is oblique it is rejected and the node becomes a leaf"""
    X, y = diagonal(1000, 0, 1)
    look = LOT(depth=3, gamma=1.0, lookahead=True, oblique=True).fit(X, y)
    assert countExtra(look.tree) == 0 and look.nLeaves() > 2
    greedy = LOT(depth=3, gamma=1.0, oblique=True).fit(X, y)
    assert countExtra(greedy.tree) == 0 and greedy.nLeaves() == 1


def testGammaThresholdOnRotatedParity():
    """depth 2: the Figure 4 tree costs 3 gamma + 4 lam, while one oblique root over two axis splits
    costs gamma + 4 lam + 1/4 (the axis test x3 <= 0 leaves error 1/4 in each child), so LOT
    returns the Figure 4 tree only for gamma < 1/8, well below the leaf bound of spec 6.7"""
    for seed in range(2):
        X, y = rotatedParity(1000, 0, seed)
        lo = LOT(depth=2, gamma=0.08, lookahead=True, oblique=True).fit(X, y)
        assert countExtra(lo.tree) == 3
        hi = LOT(depth=2, gamma=0.17, lookahead=True, oblique=True).fit(X, y)
        assert countExtra(hi.tree) <= 1
