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


def testAxis():
    for seed in range(5):
        for X, y in datasets(seed, 400):
            for depth in (2, 4):
                check(X, y, depth=depth)


def testOblique():
    for seed in range(3):
        for X, y in datasets(seed, 250):
            check(X, y, depth=2, gamma=0.0025, oblique=True, seed=seed)
    X, y = rotatedParity(200, 2, 0)
    for s in schedules.values():
        check(X, y, depth=3, oblique=True, s=s)
