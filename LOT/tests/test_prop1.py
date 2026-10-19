from lot.data import parity, rotatedParity, diagonal, orthant, trunk
from lot.tree import LOT, schedules

"""Proposition 12: R(Look) <= R(Greedy); a failure points at candidate seeding (spec 7.8 item 5)"""


def datasets(seed, n):
    """p = 5 and p = 8, so parity samples triples and trunk samples pairs too"""
    return [parity(n, 2, 3, seed), parity(n, 3, 2, seed), rotatedParity(n, 2, seed),
            diagonal(n, 2, seed), orthant(n, 3, 1, seed), trunk(n, 8, seed)]


def check(X, y, **kw):
    g = LOT(lookahead=False, **kw).fit(X, y)
    l = LOT(lookahead=True, **kw).fit(X, y)
    assert l.cost <= g.cost + 1e-12, kw


def testAxisManySeeds():
    for seed in range(10):
        for X, y in datasets(seed, 400):
            for depth in (2, 3, 4):
                for lam in (0.001, 0.02):
                    check(X, y, depth=depth, lam=lam)


def testObliqueManySeeds():
    for seed in range(6):
        for X, y in datasets(seed, 250):
            for gamma in (0.0, 0.0025):
                check(X, y, depth=2, gamma=gamma, oblique=True, seed=seed)


def testObliqueDepth3():
    """reaches level 2; slow, so fewer seeds"""
    for seed in range(2):
        for X, y in datasets(seed, 200):
            check(X, y, depth=3, lam=0.01, oblique=True, seed=seed)


def testObliqueSchedules():
    for X, y in datasets(0, 200):
        for s in schedules.values():
            check(X, y, depth=3, oblique=True, s=s)
