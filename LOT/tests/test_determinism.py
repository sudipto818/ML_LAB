import os
import subprocess
import sys
import numpy as np
from lot.data import parity, trunk
from lot.tree import LOT, schedules

"""determinism, spec 6.3 and 7.6"""


def testSameSeedSameTreeAllVariants():
    """p = 10 samples pairs and triples, so the seeded draw is exercised"""
    X, y = trunk(300, 10, 0)
    for la in (False, True):
        for ob in (False, True):
            a = LOT(depth=3, lookahead=la, oblique=ob, seed=5).fit(X, y)
            b = LOT(depth=3, lookahead=la, oblique=ob, seed=5).fit(X, y)
            assert a.tree == b.tree and a.cost == b.cost


def testCandidatesDependOnlyOnNode():
    """Proposition 12 needs this"""
    X, y = trunk(300, 10, 0)
    m = LOT(depth=4, oblique=True, s=schedules["flat"], seed=3).fit(X, y)
    idx = np.arange(0, 300, 3)
    M1, meta1 = m.candidates(idx, 3)
    M2, meta2 = m.candidates(idx.copy(), 3)
    assert np.array_equal(M1, M2) and meta1 == meta2
    assert m.candidates(idx, 2)[1] != meta1
    m.seed = 4
    assert m.candidates(idx, 3)[1] != meta1


def fitInFreshPython(hashSeed):
    code = ("from lot.data import trunk; from lot.tree import LOT; X, y = trunk(200, 10, 0); "
            "print(repr(LOT(depth=2, lookahead=True, oblique=True, seed=1).fit(X, y).tree))")
    env = dict(os.environ, PYTHONHASHSEED=str(hashSeed), PYTHONDONTWRITEBYTECODE="1")
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    return subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True,
                          text=True, check=True).stdout


def testIndependentOfPythonHash():
    """spec 7.6: seed from crc32, never hash(), which changes per process"""
    assert fitInFreshPython(1) == fitInFreshPython(2)


def testDifferentDataDifferentTree():
    """the equal trees above are not trivially equal"""
    a = LOT(depth=3).fit(*parity(400, 2, 3, 0))
    b = LOT(depth=3).fit(*parity(400, 2, 3, 1))
    assert a.tree != b.tree
