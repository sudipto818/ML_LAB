import itertools
import math
import zlib
import numpy as np

def nProjections(p, r):
    """|P_r| = C(p, r) 2^(r-1): the first sign is fixed, since a and -a give the same split"""
    return math.comb(p, r) * 2 ** (r - 1)


def sampleProjections(p, s, m, idx, level, seed):
    """for r = 2..s: all of P_r if |P_r| <= m, else m distinct draws seeded only by node, level, seed
    a projection is (support, signs): ((0, 3), (1, -1)) is x1 - x4"""
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
    """a^T x summed left to right; fit and predict share this order, so a sample on t agrees"""
    z = Xs[idx, sup[0]] * sg[0]
    for c in range(1, len(sup)):
        z = z + Xs[idx, sup[c]] * sg[c]
    return z


def obliqueCandidates(Xs, idx, projs, q):
    """q node-quantile thresholds per projection, all projections at once (spec 7.6 step 3)"""
    if len(projs) == 0:
        return np.zeros((len(idx), 0), dtype=bool), []
    qs = []
    for k in range(1, q + 1):
        qs.append(k / (q + 1))
    K = len(projs)
    r = max(len(sup) for sup, sg in projs)
    """pad with coefficient 0, which leaves z bit-identical to project()"""
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
    """method lower: thresholds are data values"""
    T = np.quantile(Z, qs, axis=0, method="lower")
    """drop t = max (not a split) and repeated quantiles"""
    ok = T < Z.max(axis=0)
    ok[1:] &= T[1:] != T[:-1]
    T = T.T
    ok = ok.T.reshape(-1)
    M = (Z[:, :, None] <= T[None, :, :]).reshape(len(idx), K * q)[:, ok]
    meta = []
    for f in np.flatnonzero(ok):
        sup, sg = projs[f // q]
        meta.append((sup, sg, T[f // q, f % q]))
    return M, meta