from lot.binarize import standardize, axisThresholds, binarize
from lot.oblique import sampleProjections, obliqueCandidates, project
import numpy as np

def oneHot(y, C):
    return np.eye(C)[y]


def entropy(counts):
    counts = np.atleast_2d(counts).astype(float)
    n = counts.sum(axis=1, keepdims=True)
    safeN = np.where(n > 0, n, 1)
    p = counts / safeN
    """0 log 0 = 0"""
    safeP = np.where(p > 0, p, 1)
    h = -(p * np.log2(safeP)).sum(axis=1)
    return h


def leafCost(Y, idx, lam):
    """u(D) = lam + (|D| - max_c n_c) / N, with N the root size, not |D|"""
    N = len(Y)
    cnt = Y[idx].sum(axis=0)
    pred = int(np.argmax(cnt))
    u = lam + (len(idx) - cnt[pred]) / N
    return u, pred


def gains(M, Yd):
    """information gain of every column of M (True = left) at once"""
    n = len(Yd)
    tot = Yd.sum(axis=0)
    L = M.T.astype(float) @ Yd
    R = tot[None, :] - L
    nl = L.sum(axis=1)
    nr = R.sum(axis=1)
    g = entropy(tot)[0] - (nl / n) * entropy(L) - (nr / n) * entropy(R)
    return g


def pick(score, cs):
    """argmin, rounded to 12 digits; a tie goes to the lower c, then the lower index"""
    s = np.round(score, 12)
    best = 0
    for f in range(1, len(s)):
        if s[f] < s[best] or (s[f] == s[best] and cs[f] < cs[best]):
            best = f
    return best


"""s(level) schedules of spec 6.8"""
schedules = {"flat": (3, 3, 3, 3), "rootHeavy": (3, 3, 2, 1), "tailHeavy": (1, 1, 3, 3)}


class LOT:
    def __init__(self, depth=4, lam=0.005, gamma=0.0, lookahead=False, oblique=False,
                 qAx=5, s=3, m=30, q=3, nMin=5, seed=0, numeric=None, screen=0, axisLocal=False):
        """s is one int or a per-level schedule, s[0] at the root
        numeric: the features projections may use (spec 9: numeric only); screen, axisLocal: ablations of 8.4"""
        self.depth = depth
        self.lam = lam
        self.gamma = gamma
        self.lookahead = lookahead
        self.oblique = oblique
        self.qAx = qAx
        self.s = (s,) * depth if isinstance(s, int) else tuple(s)
        assert len(self.s) >= depth, "the schedule needs an s for every level"
        self.m = m
        self.q = q
        self.nMin = nMin
        self.seed = seed
        self.numeric = numeric
        self.screen = screen
        self.axisLocal = axisLocal

    def fit(self, X, y):
        self.Xs, self.mu, self.sd = standardize(X)
        self.N = len(X)
        self.C = int(y.max()) + 1
        self.Y = oneHot(y, self.C)
        ax = axisThresholds(self.Xs, self.qAx)
        self.B = binarize(self.Xs, ax)
        """a test is (support, signs, threshold); axis: ((j,), (1,), t)"""
        self.meta = []
        for (j, t) in ax:
            self.meta.append(((j,), (1,), t))
        p = self.Xs.shape[1]
        self.num = np.arange(p) if self.numeric is None else np.sort(np.array(self.numeric))
        self.axisProjs = [((j,), (1,)) for j in range(p)]
        idx = np.arange(self.N)
        if self.lookahead:
            self.cost, self.tree = self.look(idx, self.depth)
        else:
            self.cost, self.tree = self.greedy(idx, self.depth)
        return self

    def candidates(self, idx, depth):
        """axis columns first so a tie keeps the axis test; same node -> same projections (Prop. 12)"""
        if self.axisLocal:
            """threshold control: axis thresholds from this node's quantiles, like the oblique ones"""
            M, allMeta = obliqueCandidates(self.Xs, idx, self.axisProjs, self.qAx)
        else:
            M = self.B[idx]
            allMeta = self.meta
        level = self.depth - depth
        if self.oblique and self.s[level] >= 2:
            projs = sampleProjections(len(self.num), self.s[level], self.m, idx, level, self.seed)
            for f in range(len(projs)):
                projs[f] = (tuple(int(self.num[j]) for j in projs[f][0]), projs[f][1])
            Mo, metaO = obliqueCandidates(self.Xs, idx, projs, self.q)
            M = np.hstack([M, Mo])
            allMeta = allMeta + metaO
        nl = M.sum(axis=0)
        n = len(idx)
        keep = np.flatnonzero((nl >= self.nMin) & (n - nl >= self.nMin))
        meta = []
        for f in keep:
            meta.append(allMeta[f])
        return M[:, keep], meta

    def greedy(self, idx, depth):
        u, pred = leafCost(self.Y, idx, self.lam)
        """u = lam means the node is pure"""
        if depth == 0 or u - self.lam < 1e-12:
            return u, ("leaf", pred)
        M, meta = self.candidates(idx, depth)
        if len(meta) == 0:
            return u, ("leaf", pred)
        g = gains(M, self.Y[idx])
        cs = []
        for m in meta:
            cs.append(len(m[0]) - 1)
        """Algorithm 1: pick by gain alone, gamma only enters the split-or-leaf test"""
        f = pick(-g, cs)
        cl, tl = self.greedy(idx[M[:, f]], depth - 1)
        cr, tr = self.greedy(idx[~M[:, f]], depth - 1)
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
        if self.screen > 0 and len(meta) > self.screen:
            """H6 ablation: look only at the top gains, which Remark 17 warns against"""
            top = np.sort(np.argsort(-gains(M, self.Y[idx]), kind="stable")[:self.screen])
            M = M[:, top]
            meta = [meta[f] for f in top]
        cs = []
        for m in meta:
            cs.append(len(m[0]) - 1)
        """Algorithm 2: score every split by greedy completions of its children"""
        vals = np.zeros(len(meta))
        for f in range(len(meta)):
            cl, _ = self.greedy(idx[M[:, f]], depth - 1)
            cr, _ = self.greedy(idx[~M[:, f]], depth - 1)
            vals[f] = self.gamma * cs[f] + cl + cr
        f = pick(vals, cs)
        if vals[f] >= u:
            return u, ("leaf", pred)
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
        z = project(Xs, idx, sup, sg)
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
