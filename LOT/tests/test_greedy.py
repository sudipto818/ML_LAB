import numpy as np
from lot.data import parity
from lot.tree import LOT


def error(m, X, y):
    return np.mean(m.predict(X) != y)


# y is decided by the median of x1 and the other 3 features are noise: one split solves it
def test_greedy_fits_axis_aligned_data_exactly():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (600, 4))
    y = (X[:, 0] > np.median(X[:, 0])).astype(int)
    m = LOT(depth=2, lookahead=False).fit(X, y)
    assert error(m, X, y) == 0
    assert m.n_leaves() == 2
    assert m.tree[1][0] == (0,)                     # the root test uses feature x1


# at the root every single-feature test has zero gain on XOR, so Greedy's first split is a
# coin flip: with depth 2 it is right only when it happens to pick a median threshold
def test_greedy_is_unreliable_on_xor():
    errs = []
    for seed in range(8):
        X, y = parity(1000, 2, 0, seed)
        Xt, yt = parity(4000, 2, 0, 100 + seed)
        m = LOT(depth=2, lookahead=False).fit(X, y)
        errs.append(error(m, Xt, yt))
    assert np.mean(errs) > 0.10
    assert max(errs) > 0.25


# with depth 1 (a single split) nothing can beat chance on XOR
def test_depth_one_on_xor_is_chance():
    X, y = parity(2000, 2, 0, 0)
    Xt, yt = parity(4000, 2, 0, 1)
    m = LOT(depth=1, lookahead=False).fit(X, y)
    assert abs(error(m, Xt, yt) - 0.5) < 0.05


# a split must leave at least n_min samples on each side
def test_n_min_respected():
    X, y = parity(300, 2, 2, 0)
    m = LOT(depth=4, lookahead=False, n_min=40).fit(X, y)
    def walk(node, idx):
        if node[0] == "leaf":
            assert len(idx) >= 40 or len(idx) == len(y)
            return
        sup, sg, t = node[1]
        left = m.Xs[idx, sup[0]] <= t
        walk(node[2], idx[left])
        walk(node[3], idx[~left])
    walk(m.tree, np.arange(len(y)))


# same data, same settings -> same tree
def test_deterministic():
    X, y = parity(500, 2, 3, 0)
    a = LOT(depth=3).fit(X, y)
    b = LOT(depth=3).fit(X, y)
    assert a.tree == b.tree