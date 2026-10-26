import numpy as np
from lot.data import rotatedParityLabels, stratifiedFolds


def testRotatedParityLabels():
    X = np.array([[1, 1, 1, 0], [1, 1, -1, 0], [-1, -1, 1, 0], [-1, -1, -1, 0]])
    assert list(rotatedParityLabels(X)) == [0, 1, 1, 0]


def testStratifiedFolds():
    y = np.array([0] * 53 + [1] * 31 + [2] * 16)
    fold = stratifiedFolds(y, 5, 0)
    for c in range(3):
        cnt = np.bincount(fold[y == c], minlength=5)
        assert cnt.max() - cnt.min() <= 1
    sizes = np.bincount(fold, minlength=5)
    assert sizes.max() - sizes.min() <= 1
