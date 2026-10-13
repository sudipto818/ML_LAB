import numpy as np

# labels y (0..C-1) -> one-hot matrix of shape (N, C)
def one_hot(y, C):
    N = len(y)
    Y = np.zeros((N, C))
    for i in range(N):
        Y[i, y[i]] = 1
    return Y


# H = -summation (p log2 p) for every row of counts, with 0 log 0 = 0
# counts is (k, C), or a single vector of length C
def entropy(counts):
    counts = np.atleast_2d(counts).astype(float)
    n = counts.sum(axis=1, keepdims=True)
    safe_n = np.where(n > 0, n, 1)          
    p = counts / safe_n
    safe_p = np.where(p > 0, p, 1)          # log2(1) = 0, so these terms vanish
    h = -(p * np.log2(safe_p)).sum(axis=1)
    return h


# u(D) = lam + (|D| - max_c n_c(D)) / N, always divided by the root N
# Y is the one-hot matrix of the whole training set, idx are the samples in this node
def leaf_cost(Y, idx, lam):
    N = len(Y)
    cnt = Y[idx].sum(axis=0)
    pred = int(np.argmax(cnt))
    u = lam + (len(idx) - cnt[pred]) / N
    return u, pred


# information gain of every candidate test in one shot
# M  : (n, k) boolean, M[i, f] = True if sample i goes left under test f
# Yd : (n, C) one-hot labels of the samples in this node
def gains(M, Yd):
    n = len(Yd)
    tot = Yd.sum(axis=0)                    # class counts of the node
    L = M.T.astype(float) @ Yd              # (k, C) class counts going left
    R = tot[None, :] - L                    # (k, C) class counts going right
    nl = L.sum(axis=1)
    nr = R.sum(axis=1)
    g = entropy(tot)[0] - (nl / n) * entropy(L) - (nr / n) * entropy(R)
    return g