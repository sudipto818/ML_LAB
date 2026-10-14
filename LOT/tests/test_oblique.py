import numpy as np
from lot.data import diagonal, rotated_parity
from lot.binarize import standardize
from lot.oblique import n_projections, sample_projections, oblique_candidates
from lot.tree import one_hot, gains


def test_counts():
    assert n_projections(6, 2) == 6 * 5            # p(p-1)
    assert n_projections(6, 3) == 20 * 4
    assert n_projections(4, 2) == 12


def valid(projs, p):
    for (sup, sg) in projs:
        assert list(sup) == sorted(set(sup))       # sorted, no repeated feature
        assert max(sup) < p and min(sup) >= 0
        assert len(sup) == len(sg)
        assert sg[0] == 1
        for v in sg:
            assert v in (1, -1)


# small p: everything is listed, nothing is sampled
def test_enumerates_all_when_few():
    idx = np.arange(100)
    projs = sample_projections(4, 2, 30, idx, 0, 0)
    assert len(projs) == 12
    assert len(set(projs)) == 12
    valid(projs, 4)
    assert ((0, 1), (1, 1)) in projs               # x1 + x2
    assert ((2, 3), (1, -1)) in projs              # x3 - x4  (the rotated parity projections)


# large p: m distinct projections for each support size
def test_samples_m_per_size():
    idx = np.arange(100)
    projs = sample_projections(10, 3, 30, idx, 0, 0)
    assert len(projs) == 60
    assert len(set(projs)) == 60
    valid(projs, 10)
    n2 = 0
    n3 = 0
    for (sup, sg) in projs:
        if len(sup) == 2:
            n2 += 1
        else:
            n3 += 1
    assert n2 == 30 and n3 == 30


def test_s_below_2_gives_no_projections():
    assert sample_projections(10, 1, 30, np.arange(50), 0, 0) == []


def test_deterministic_and_varies_with_node_level_seed():
    a = sample_projections(10, 3, 30, np.arange(100), 2, 7)
    b = sample_projections(10, 3, 30, np.arange(100), 2, 7)
    assert a == b
    assert a != sample_projections(10, 3, 30, np.arange(100), 3, 7)      # other level
    assert a != sample_projections(10, 3, 30, np.arange(100), 2, 8)      # other run seed
    assert a != sample_projections(10, 3, 30, np.arange(1, 101), 2, 7)   # other samples


# columns follow the definition, thresholds lie inside the data, at most q per projection
def test_candidate_columns():
    X, y = rotated_parity(600, 2, 0)
    Xs, mu, sd = standardize(X)
    idx = np.arange(0, 600, 2)
    projs = sample_projections(6, 3, 30, idx, 0, 0)
    M, meta = oblique_candidates(Xs, idx, projs, 3)
    assert M.shape == (len(idx), len(meta))
    assert M.dtype == bool
    assert len(meta) <= 3 * len(projs)
    for f in range(0, len(meta), 7):
        sup, sg, t = meta[f]
        z = Xs[np.ix_(idx, sup)] @ np.array(sg, dtype=float)
        assert np.array_equal(M[:, f], z <= t)
        assert z.min() <= t < z.max()


# on y = 1[x1 + x2 > 0] the projection x1 + x2 at the median threshold is an (almost) perfect split
def test_finds_the_diagonal_split():
    X, y = diagonal(2000, 0, 0)
    Xs, mu, sd = standardize(X)
    idx = np.arange(2000)
    projs = sample_projections(2, 2, 30, idx, 0, 0)
    assert len(projs) == 2                          # x1 + x2 and x1 - x2
    M, meta = oblique_candidates(Xs, idx, projs, 3)
    g = gains(M, one_hot(y, 2))
    best = int(np.argmax(g))
    assert meta[best][0] == (0, 1) and meta[best][1] == (1, 1)
    assert g[best] > 0.9