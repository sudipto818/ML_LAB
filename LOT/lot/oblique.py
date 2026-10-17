import itertools
import math
import zlib
import numpy as np

def nProjections(p, r):
    """|P_r| = C(p, r) * 2^(r-1): choose r features, signs +-1 each, first sign fixed to +1
    (a and -a describe the same split with the sides swapped, so we only count one of them)"""
    return math.comb(p, r) * 2 ** (r - 1)


def sampleProjections(p, s, m, idx, level, seed):
    """sparse projections with support sizes r = 2, ..., s, at most m for each size r
      - if |P_r| <= m we list every projection of size r (nothing is left to chance)
      - otherwise we draw m distinct ones, with a seed that depends only on the node's samples,
        the level and the run seed, so the same node always gets the same projections
    a projection is (support, signs), e.g. ((0, 3), (1, -1)) means x1 - x4"""
    out = []
    rng = np.random.default_rng([zlib.crc32(idx.tobytes()), level, seed])
    for r in range(2, s + 1):
        if r > p:
            break
        if nProjections(p, r) <= m:
            for sup in itertools.combinations(range(p), r):
                for sg in itertools.product((1, -1), repeat=r - 1):
                    out.append((sup, (1,) + sg))
        else:
            seen = set()
            while len(seen) < m:
                sup = tuple(sorted(rng.choice(p, r, replace=False).tolist()))
                sg = (1,) + tuple(rng.choice((1, -1), r - 1).tolist())
                seen.add((sup, sg))
            out.extend(sorted(seen))
    return out


def project(Xs, idx, sup, sg):
    """z = a^T x for the samples idx, added left to right over the support
    fit and predict both use this order, so a sample sitting exactly on a threshold
    (thresholds are data values) goes to the same side in both"""
    z = Xs[idx, sup[0]] * sg[0]
    for c in range(1, len(sup)):
        z = z + Xs[idx, sup[c]] * sg[c]
    return z


def obliqueCandidates(Xs, idx, projs, q):
    """for every projection z = Xs[idx, support] @ signs take q thresholds at the quantiles
    1/(q+1), ..., q/(q+1) of z, using this node's samples only
    returns M (n x k boolean, True = goes left) and meta, a list of (support, signs, threshold)
    all projections at once (spec 7.6 step 3): Z is n x K, one column per projection"""
    if len(projs) == 0:
        return np.zeros((len(idx), 0), dtype=bool), []
    qs = []
    for k in range(1, q + 1):
        qs.append(k / (q + 1))
    K = len(projs)
    r = max(len(sup) for sup, sg in projs)
    """supports padded to r with coefficient 0: adding 0 * x leaves z exactly as project() gives it"""
    S = np.zeros((K, r), dtype=int)
    G = np.zeros((K, r))
    for f in range(K):
        sup, sg = projs[f]
        S[f, :len(sup)] = sup
        G[f, :len(sg)] = sg
    Xi = Xs[idx]
    Z = Xi[:, S[:, 0]] * G[:, 0]
    for c in range(1, r):
        Z = Z + Xi[:, S[:, c]] * G[:, c]
    """T[i, f] = i-th quantile of projection f; method lower keeps them real data values, in increasing order"""
    T = np.quantile(Z, qs, axis=0, method="lower")
    """keep t < max (t = max sends everything left) and drop a quantile equal to the one before it"""
    ok = T < Z.max(axis=0)
    ok[1:] &= T[1:] != T[:-1]
    """column order: projection by projection, thresholds increasing, as listed in projs"""
    T = T.T
    ok = ok.T.reshape(-1)
    M = (Z[:, :, None] <= T[None, :, :]).reshape(len(idx), K * q)[:, ok]
    meta = []
    for f in np.flatnonzero(ok):
        sup, sg = projs[f // q]
        meta.append((sup, sg, T[f // q, f % q]))
    return M, meta