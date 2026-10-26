import numpy as np
from lot.data import rotatedParity, trunk
from lot.tree import LOT


def supports(node, out):
    if node[0] == "split":
        out.append(node[1][0])
        supports(node[2], out)
        supports(node[3], out)
    return out


def testNumericRestrictsProjections():
    X, y = trunk(400, 8, 0)
    m = LOT(depth=3, lookahead=True, oblique=True, numeric=[1, 3, 4, 6]).fit(X, y)
    sup = supports(m.tree, [])
    assert any(len(s) > 1 for s in sup)
    for s in sup:
        assert len(s) == 1 or set(s) <= {1, 3, 4, 6}


def testScreeningLosesRotatedParity():
    """Remark 17: the right root split has zero gain, so top-10 screening drops it"""
    X, y = rotatedParity(1000, 0, 0)
    Xt, yt = rotatedParity(20000, 0, 100)
    full = LOT(depth=2, lookahead=True, oblique=True).fit(X, y)
    screened = LOT(depth=2, lookahead=True, oblique=True, screen=10).fit(X, y)
    assert np.mean(full.predict(Xt) != yt) < 0.05
    assert np.mean(screened.predict(Xt) != yt) > 0.2


def testAxisLocalThresholds():
    """the global thresholds at the root, the node's own quantiles below it"""
    X, y = trunk(500, 4, 0)
    g = LOT(depth=3).fit(X, y)
    l = LOT(depth=3, axisLocal=True).fit(X, y)
    assert g.candidates(np.arange(500), 3)[1] == l.candidates(np.arange(500), 3)[1]
    node = np.flatnonzero(g.Xs[:, 0] <= 0)
    meta = l.candidates(node, 2)[1]
    assert meta != g.candidates(node, 2)[1]
    for sup, sg, t in meta:
        assert t in g.Xs[node, sup[0]]
