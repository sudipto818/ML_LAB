import numpy as np

def parityLabels(X, r):
    """y = XOR of 1[x_j > 0], j < r"""
    n = len(X)
    y = np.zeros(n, dtype=int)
    for i in range(n):
        cnt = 0
        for j in range(r):
            if X[i, j] > 0:
                cnt += 1
        y[i] = cnt % 2
    return y


def rotatedParityLabels(X):
    """y = 1[x1 + x2 > 0] XOR 1[x3 - x4 > 0]"""
    n = len(X)
    y = np.zeros(n, dtype=int)
    for i in range(n):
        a = 0
        b = 0
        if X[i, 0] + X[i, 1] > 0:
            a = 1
        if X[i, 2] - X[i, 3] > 0:
            b = 1
        y[i] = a ^ b
    return y


def orthantLabels(X, p):
    """y = orthant of the first p features, 0..2^p-1"""
    n = len(X)
    y = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(p):
            if X[i, j] > 0:
                y[i] += 2 ** j
    return y


def diagonalLabels(X):
    """y = 1[x1 + x2 > 0]"""
    n = len(X)
    y = np.zeros(n, dtype=int)
    for i in range(n):
        if X[i, 0] + X[i, 1] > 0:
            y[i] = 1
    return y

def parity(n, r, noise=0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, r + noise))
    y = parityLabels(X, r)
    return X, y


def rotatedParity(n, noise=0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, 4 + noise))
    y = rotatedParityLabels(X)
    return X, y


def orthant(n, p, noise=0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, p + noise))
    y = orthantLabels(X, p)
    return X, y


def diagonal(n, noise=0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, 2 + noise))
    y = diagonalLabels(X)
    return X, y


def trunk(n, p, seed=0):
    """Trunk: N(+mu, I) vs N(-mu, I), mu_j = 1/sqrt(j)"""
    rng = np.random.default_rng(seed)
    mu = 1 / np.sqrt(np.arange(1, p + 1))
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, p))
    for i in range(n):
        if y[i] == 1:
            X[i] += mu
        else:
            X[i] -= mu
    return X, y


def ringnorm(n, seed=0):
    """Breiman's ringnorm, p = 20: N(0, 4I) vs N(a, I), a_j = 2/sqrt(20)"""
    rng = np.random.default_rng(seed)
    p = 20
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, p))
    for i in range(n):
        if y[i] == 0:
            X[i] *= 2
        else:
            X[i] += 2 / np.sqrt(p)
    return X, y


def loadCsv(path):
    """header row, numeric features, class 0..C-1 in the last column"""
    A = np.loadtxt(path, delimiter=",", skiprows=1)
    return A[:, :-1], A[:, -1].astype(int)


def subsample(X, y, n, seed=0):
    if len(y) <= n:
        return X, y
    idx = np.sort(np.random.default_rng(seed).permutation(len(y))[:n])
    return X[idx], y[idx]


def stratifiedFolds(y, k, seed):
    """fold id per sample: each class shuffled, then dealt round-robin, continuing across classes"""
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(y), dtype=int)
    start = 0
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        fold[idx] = (start + np.arange(len(idx))) % k
        start += len(idx)
    return fold