import itertools
import math
import zlib
import numpy as np

# |P_r| = C(p, r) * 2^(r-1): choose r features, signs +-1 each, first sign fixed to +1
# (a and -a describe the same split with the sides swapped, so we only count one of them)
def n_projections(p, r):
    return math.comb(p, r) * 2 ** (r - 1)


# sparse projections with support sizes r = 2, ..., s, at most m for each size r
#   - if |P_r| <= m we list every projection of size r (nothing is left to chance)
#   - otherwise we draw m distinct ones, with a seed that depends only on the node's samples,
#     the level and the run seed, so the same node always gets the same projections
# a projection is (support, signs), e.g. ((0, 3), (1, -1)) means x1 - x4
def sample_projections(p, s, m, idx, level, seed):
    out = []
    rng = np.random.default_rng([zlib.crc32(idx.tobytes()), level, seed])
    for r in range(2, s + 1):
        if r > p:
            break
        if n_projections(p, r) <= m:
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


# for every projection z = Xs[idx, support] @ signs take q thresholds at the quantiles
# 1/(q+1), ..., q/(q+1) of z, using this node's samples only
# returns M (n x k boolean, True = goes left) and meta, a list of (support, signs, threshold)
def oblique_candidates(Xs, idx, projs, q):
    qs = []
    for k in range(1, q + 1):
        qs.append(k / (q + 1))
    cols = []
    meta = []
    for (sup, sg) in projs:
        z = Xs[np.ix_(idx, sup)] @ np.array(sg, dtype=float)
        ts = np.unique(np.quantile(z, qs, method="lower"))
        mx = z.max()
        for t in ts:
            if t < mx:
                cols.append(z <= t)
                meta.append((sup, sg, t))
    if len(cols) == 0:
        return np.zeros((len(idx), 0), dtype=bool), meta
    return np.stack(cols, axis=1), meta