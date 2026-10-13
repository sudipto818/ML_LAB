import numpy as np
from lot.data import (parity, rotated_parity, orthant, diagonal, trunk,parity_labels, rotated_parity_labels)


# XOR rule on 10 hand-picked points (r = 2)
def test_parity_rule():
    X = np.array([[0.3, 0.7], [0.3, -0.7], [-0.3, 0.7], [-0.3, -0.7],
                  [0.9, 0.1], [0.9, -0.1], [-0.9, 0.1], [-0.9, -0.1],
                  [0.01, 0.02], [-0.01, 0.02]])
    expected = [0, 1, 1, 0, 0, 1, 1, 0, 0, 1]
    y = parity_labels(X, 2)
    for i in range(10):
        assert y[i] == expected[i]


# rotated rule on 10 hand-picked points: y = 1[x1+x2>0] XOR 1[x3-x4>0]
def test_rotated_parity_rule():
    X = np.array([[1, 1, 1, 0], [1, 1, -1, 0], [-1, -1, 1, 0], [-1, -1, -1, 0],
                  [0.5, -0.2, 0, -1], [0.2, -0.5, 0, 1], [0.2, -0.5, 0, -1],
                  [0.5, -0.2, 0, 1], [-0.3, 0.1, 0.1, 0.5], [-0.3, 0.1, 0.5, 0.1]])
    expected = [0, 1, 1, 0, 0, 0, 1, 1, 0, 1]
    y = rotated_parity_labels(X)
    for i in range(10):
        assert y[i] == expected[i]


# shapes, labels in {0,1}, roughly balanced classes
def check_binary(X, y, n, p):
    assert X.shape == (n, p)
    assert y.shape == (n,)
    assert set(np.unique(y)) == {0, 1}
    assert abs(y.mean() - 0.5) < 0.05


def test_binary_generators():
    n = 4000
    X, y = parity(n, 2, 3, 0)
    check_binary(X, y, n, 5)
    X, y = parity(n, 3, 0, 0)
    check_binary(X, y, n, 3)
    X, y = rotated_parity(n, 2, 0)
    check_binary(X, y, n, 6)
    X, y = diagonal(n, 2, 0)
    check_binary(X, y, n, 4)
    X, y = trunk(n, 5, 0)
    check_binary(X, y, n, 5)


# orthant has 2^p classes (not binary), each near 1/2^p
def test_orthant():
    n = 8000
    X, y = orthant(n, 3, 2, 0)
    assert X.shape == (n, 5)
    assert set(np.unique(y)) == set(range(8))
    for c in range(8):
        frac = np.mean(y == c)
        assert abs(frac - 1 / 8) < 0.03


# same seed gives same data, different seed gives different data
def test_seed():
    X1, y1 = rotated_parity(100, 2, 7)
    X2, y2 = rotated_parity(100, 2, 7)
    X3, y3 = rotated_parity(100, 2, 8)
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2)
    assert not np.array_equal(X1, X3)


# class 1 is centred at +mu (all positive), class 0 at -mu
def test_trunk_means():
    X, y = trunk(20000, 4, 0)
    m1 = X[y == 1].mean(axis=0)
    m0 = X[y == 0].mean(axis=0)
    for j in range(4):
        assert m1[j] > 0
        assert m0[j] < 0