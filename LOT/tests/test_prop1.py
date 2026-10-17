from lot.data import parity, rotatedParity, diagonal, orthant, trunk
from lot.tree import LOT, schedules

"""Proposition 12: R(Look) <= R(Greedy) on the training objective, for the same data, hyperparameters
and switch O. A violation is a bug, most likely in the seeding of candidates (spec 7.8, item 5)
parity(n, 2, 3) has p = 5, so triples are sampled; trunk(n, 8) has p = 8, so pairs are sampled too"""


def datasets(seed, n):
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
    """depth 3 also reaches level 2; each Look fit here takes seconds, so fewer seeds"""
    for seed in range(2):
        for X, y in datasets(seed, 200):
            check(X, y, depth=3, lam=0.01, oblique=True, seed=seed)


def testObliqueSchedules():
    for X, y in datasets(0, 200):
        for s in schedules.values():
            check(X, y, depth=3, oblique=True, s=s)
