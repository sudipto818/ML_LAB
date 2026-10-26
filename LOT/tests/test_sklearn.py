import numpy as np
import pytest
from lot.data import trunk, ringnorm
from lot.tree import LOT

sklearnTree = pytest.importorskip("sklearn.tree")

"""spec 7.8 item 6: with O = 0 and lam -> 0, Greedy reaches the training error of an independent
entropy tree on the same binary columns. Quantile columns put equal counts left, so exact gain ties
happen; sklearn breaks them by a random feature order, ours by lowest index, so we must match one order"""


def testGreedyMatchesSklearnEntropyTree():
    for X, y in [trunk(800, 6, 0), trunk(800, 10, 1), ringnorm(800, 2)]:
        for depth in (2, 3, 4):
            m = LOT(depth=depth, lam=1e-9).fit(X, y)
            errs = set()
            for rs in range(20):
                sk = sklearnTree.DecisionTreeClassifier(criterion="entropy", max_depth=depth, min_samples_leaf=5,
                                                        random_state=rs)
                errs.add(np.mean(sk.fit(m.B, y).predict(m.B) != y))
            assert np.mean(m.predict(X) != y) in errs
