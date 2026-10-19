import numpy as np
from lot.data import parity
from lot.tree import LOT


def error(m, X, y):
    return np.mean(m.predict(X) != y)


def testGreedyFitsAxisAlignedDataExactly():
    """y depends on x1 only, so one split is exact"""
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (600, 4))
    y = (X[:, 0] > np.median(X[:, 0])).astype(int)
    m = LOT(depth=2, lookahead=False).fit(X, y)
    assert error(m, X, y) == 0
    assert m.nLeaves() == 2
    assert m.tree[1][0] == (0,)


def testGreedyIsUnreliableOnXor():
    """every root axis test has zero gain on XOR, so Greedy's root is a coin flip"""
    errs = []
    for seed in range(8):
        X, y = parity(1000, 2, 0, seed)
        Xt, yt = parity(4000, 2, 0, 100 + seed)
        m = LOT(depth=2, lookahead=False).fit(X, y)
        errs.append(error(m, Xt, yt))
    assert np.mean(errs) > 0.10
    assert max(errs) > 0.25


def testDepthOneOnXorIsChance():
    X, y = parity(2000, 2, 0, 0)
    Xt, yt = parity(4000, 2, 0, 1)
    m = LOT(depth=1, lookahead=False).fit(X, y)
    assert abs(error(m, Xt, yt) - 0.5) < 0.05


def testNMinRespected():
    X, y = parity(300, 2, 2, 0)
    m = LOT(depth=4, lookahead=False, nMin=40).fit(X, y)
    def walk(node, idx):
        if node[0] == "leaf":
            assert len(idx) >= 40 or len(idx) == len(y)
            return
        sup, sg, t = node[1]
        left = m.Xs[idx, sup[0]] <= t
        walk(node[2], idx[left])
        walk(node[3], idx[~left])
    walk(m.tree, np.arange(len(y)))


def testDeterministic():
    X, y = parity(500, 2, 3, 0)
    a = LOT(depth=3).fit(X, y)
    b = LOT(depth=3).fit(X, y)
    assert a.tree == b.tree