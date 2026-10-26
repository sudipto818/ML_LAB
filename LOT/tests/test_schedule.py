from lot.data import rotatedParity, diagonal
from lot.tree import LOT, schedules
from lot.metrics import countExtra


def splitsByLevel(node, level, out):
    """(level, ||a||_0) of every split"""
    if node[0] == "split":
        out.append((level, len(node[1][0])))
        splitsByLevel(node[2], level + 1, out)
        splitsByLevel(node[3], level + 1, out)
    return out


def testScheduleBoundsEveryLevel():
    X, y = rotatedParity(600, 2, 0)
    for s in schedules.values():
        m = LOT(depth=4, oblique=True, s=s).fit(X, y)
        for level, nnz in splitsByLevel(m.tree, 0, []):
            assert nnz <= s[level]


def testLargeGammaRemovesObliqueSplits():
    """Look falls back to axis splits; Greedy picks by gain alone (Algorithm 1), so it ends at a leaf"""
    X, y = diagonal(1000, 0, 1)
    look = LOT(depth=3, gamma=1.0, lookahead=True, oblique=True).fit(X, y)
    assert countExtra(look.tree) == 0 and look.nLeaves() > 2
    greedy = LOT(depth=3, gamma=1.0, oblique=True).fit(X, y)
    assert greedy.nLeaves() == 1


def testGammaThresholdOnRotatedParity():
    """Figure 4 tree: 3 gamma + 4 lam; oblique root over two axis splits: gamma + 4 lam + 1/4,
    so Figure 4 needs gamma < 1/8, stricter than the leaf bound of spec 6.7"""
    for seed in range(2):
        X, y = rotatedParity(1000, 0, seed)
        assert countExtra(LOT(depth=2, gamma=0.08, lookahead=True, oblique=True).fit(X, y).tree) == 3
        assert countExtra(LOT(depth=2, gamma=0.17, lookahead=True, oblique=True).fit(X, y).tree) <= 1
