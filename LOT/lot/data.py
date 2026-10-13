import numpy as np

# y = XOR of 1[x_j > 0] over the first r features
def parity_labels(X, r):
    n = len(X)
    y = np.zeros(n, dtype=int)
    for i in range(n):
        cnt = 0
        for j in range(r):
            if X[i, j] > 0:
                cnt += 1
        y[i] = cnt % 2
    return y


# y = 1[x1 + x2 > 0] XOR 1[x3 - x4 > 0]
def rotated_parity_labels(X):
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


# class = index of the orthant of the first p features, in 0..2^p-1
def orthant_labels(X, p):
    n = len(X)
    y = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(p):
            if X[i, j] > 0:
                y[i] += 2 ** j
    return y


# y = 1[x1 + x2 > 0]
def diagonal_labels(X):
    n = len(X)
    y = np.zeros(n, dtype=int)
    for i in range(n):
        if X[i, 0] + X[i, 1] > 0:
            y[i] = 1
    return y

# r informative features, `noise` extra features with no information
def parity(n, r, noise=0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, r + noise))
    y = parity_labels(X, r)
    return X, y


# first 4 features matter, the rest are noise
def rotated_parity(n, noise=0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, 4 + noise))
    y = rotated_parity_labels(X)
    return X, y


# 2^p classes, one per orthant of the first p features
def orthant(n, p, noise=0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, p + noise))
    y = orthant_labels(X, p)
    return X, y


# two classes separated by the line x1 + x2 = 0
def diagonal(n, noise=0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, 2 + noise))
    y = diagonal_labels(X)
    return X, y


# Trunk problem: two Gaussians with means +mu and -mu, mu_j = 1/sqrt(j), identity covariance
def trunk(n, p, seed=0):
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