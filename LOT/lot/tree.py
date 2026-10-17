from lot.binarize import standardize, axisThresholds, binarize
from lot.oblique import sampleProjections, obliqueCandidates
import numpy as np

def oneHot(y, C):
    """labels y (0..C-1) -> one-hot matrix of shape (N, C)"""
    N = len(y)
    Y = np.zeros((N, C))
    for i in range(N):
        Y[i, y[i]] = 1
    return Y


def entropy(counts):
    """H = -summation (p log2 p) for every row of counts, with 0 log 0 = 0
    counts is (k, C), or a single vector of length C"""
    counts = np.atleast_2d(counts).astype(float)
    n = counts.sum(axis=1, keepdims=True)
    safeN = np.where(n > 0, n, 1)          
    p = counts / safeN
    """log2(1) = 0, so these terms vanish"""
    safeP = np.where(p > 0, p, 1)
    h = -(p * np.log2(safeP)).sum(axis=1)
    return h


def leafCost(Y, idx, lam):
    """u(D) = lam + (|D| - max_c n_c(D)) / N, always divided by the root N
    Y is the one-hot matrix of the whole training set, idx are the samples in this node"""
    N = len(Y)
    cnt = Y[idx].sum(axis=0)
    pred = int(np.argmax(cnt))
    u = lam + (len(idx) - cnt[pred]) / N
    return u, pred


def gains(M, Yd):
    """information gain of every candidate test in one shot
    M  : (n, k) boolean, M[i, f] = True if sample i goes left under test f
    Yd : (n, C) one-hot labels of the samples in this node"""
    n = len(Yd)
    """class counts of the node"""
    tot = Yd.sum(axis=0)
    """(k, C) class counts going left"""
    L = M.T.astype(float) @ Yd
    """(k, C) class counts going right"""
    R = tot[None, :] - L
    nl = L.sum(axis=1)
    nr = R.sum(axis=1)
    g = entropy(tot)[0] - (nl / n) * entropy(L) - (nr / n) * entropy(R)
    return g


def pick(score, cs):
    """lowest score wins; scores are rounded to 12 digits so ties are stable,
    a tie goes to the lower cost, and then to the lower index"""
    s = np.round(score, 12)
    best = 0
    for f in range(1, len(s)):
        if s[f] < s[best] or (s[f] == s[best] and cs[f] < cs[best]):
            best = f
    return best


class LOT:
    def __init__(self, depth=4, lam=0.005, gamma=0.0, lookahead=False, oblique=False,
                 qAx=5, s=3, m=30, q=3, nMin=5, seed=0):
        """lookahead=False -> greedy, lookahead=True -> look (LicketySPLIT while only axis tests are used)
        oblique=True adds sparse projections: up to s non-zeros, m per support size, q thresholds each"""
        self.depth = depth
        self.lam = lam
        self.gamma = gamma
        self.lookahead = lookahead
        self.oblique = oblique
        self.qAx = qAx
        self.s = s
        self.m = m
        self.q = q
        self.nMin = nMin
        self.seed = seed

    def fit(self, X, y):
        self.Xs, self.mu, self.sd = standardize(X)
        self.N = len(X)
        self.C = int(y.max()) + 1
        self.Y = oneHot(y, self.C)
        ax = axisThresholds(self.Xs, self.qAx)
        self.B = binarize(self.Xs, ax)
        """a test is stored as (features used, signs, threshold); an axis test uses one feature"""
        self.meta = []
        for (j, t) in ax:
            self.meta.append(((j,), (1,), t))
        idx = np.arange(self.N)
        if self.lookahead:
            self.cost, self.tree = self.look(idx, self.depth)
        else:
            self.cost, self.tree = self.greedy(idx, self.depth)
        return self

    def candidates(self, idx, depth):
        """valid tests for the samples idx: at least nMin samples on each side
        axis columns come first, oblique ones after, so on a tie the lower index keeps the axis test
        level = self.depth - depth is the same whether greedy or look reaches this node, so both
        see the same projections (Proposition 12 needs that)"""
        M = self.B[idx]
        allMeta = self.meta
        if self.oblique:
            projs = sampleProjections(self.Xs.shape[1], self.s, self.m, idx, self.depth - depth, self.seed)
            Mo, metaO = obliqueCandidates(self.Xs, idx, projs, self.q)
            M = np.hstack([M, Mo])
            allMeta = self.meta + metaO
        nl = M.sum(axis=0)
        n = len(idx)
        keep = np.flatnonzero((nl >= self.nMin) & (n - nl >= self.nMin))
        meta = []
        for f in keep:
            meta.append(allMeta[f])
        return M[:, keep], meta

    def greedy(self, idx, depth):
        u, pred = leafCost(self.Y, idx, self.lam)
        """out of depth, or the node is already pure"""
        if depth == 0 or u - self.lam < 1e-12:
            return u, ("leaf", pred)
        M, meta = self.candidates(idx, depth)
        if len(meta) == 0:
            return u, ("leaf", pred)
        g = gains(M, self.Y[idx])
        cs = []
        for m in meta:
            cs.append(len(m[0]) - 1)
        f = pick(-g, cs)
        cl, tl = self.greedy(idx[M[:, f]], depth - 1)
        cr, tr = self.greedy(idx[~M[:, f]], depth - 1)
        """Objective cost of candiate split via recursion"""
        v = self.gamma * cs[f] + cl + cr
        if v < u:
            return v, ("split", meta[f], tl, tr)
        return u, ("leaf", pred)

    def look(self, idx, depth):
        u, pred = leafCost(self.Y, idx, self.lam)
        if depth == 0 or u - self.lam < 1e-12:
            return u, ("leaf", pred)
        M, meta = self.candidates(idx, depth)
        if len(meta) == 0:
            return u, ("leaf", pred)
        cs = []
        for m in meta:
            cs.append(len(m[0]) - 1)
        vals = np.zeros(len(meta))
        for f in range(len(meta)):
            """children are scored by Greedy"""
            cl, _ = self.greedy(idx[M[:, f]], depth - 1)
            cr, _ = self.greedy(idx[~M[:, f]], depth - 1)
            vals[f] = self.gamma * cs[f] + cl + cr
        f = pick(vals, cs)
        if vals[f] >= u:
            return u, ("leaf", pred)
        """the chosen split recurses with Look"""
        cl, tl = self.look(idx[M[:, f]], depth - 1)
        cr, tr = self.look(idx[~M[:, f]], depth - 1)
        return self.gamma * cs[f] + cl + cr, ("split", meta[f], tl, tr)

    def predict(self, X):
        Xs = (X - self.mu) / self.sd
        out = np.zeros(len(X), dtype=int)
        self.walk(self.tree, Xs, np.arange(len(X)), out)
        return out

    def walk(self, node, Xs, idx, out):
        if node[0] == "leaf":
            out[idx] = node[1]
            return
        sup, sg, t = node[1]
        z = Xs[np.ix_(idx, sup)] @ np.array(sg, dtype=float)
        left = z <= t
        self.walk(node[2], Xs, idx[left], out)
        self.walk(node[3], Xs, idx[~left], out)

    def nLeaves(self, node=None):
        if node is None:
            node = self.tree
        if node[0] == "leaf":
            return 1
        return self.nLeaves(node[2]) + self.nLeaves(node[3])

    def show(self, node=None, ind=""):
        if node is None:
            node = self.tree
        if node[0] == "leaf":
            print(ind + "-> class " + str(node[1]))
            return
        sup, sg, t = node[1]
        expr = ""
        for j, s in zip(sup, sg):
            expr += ("+" if s > 0 else "-") + "x" + str(j + 1) + " "
        print(ind + expr + "<= " + str(round(float(t), 3)))
        self.show(node[2], ind + "    ")
        self.show(node[3], ind + "    ")
