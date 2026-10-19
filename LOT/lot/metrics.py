import numpy as np

def errorRate(yTrue, yPred):
    wrong = 0
    for i in range(len(yTrue)):
        if yTrue[i] != yPred[i]:
            wrong += 1
    return wrong / len(yTrue)


def countLeaves(node):
    if node[0] == "leaf":
        return 1
    return countLeaves(node[2]) + countLeaves(node[3])


def countSplits(node):
    if node[0] == "leaf":
        return 0
    return 1 + countSplits(node[2]) + countSplits(node[3])


def countNnz(node):
    """sum of ||a||_0"""
    if node[0] == "leaf":
        return 0
    return len(node[1][0]) + countNnz(node[2]) + countNnz(node[3])


def countExtra(node):
    """sum of c(a) = ||a||_0 - 1"""
    if node[0] == "leaf":
        return 0
    return len(node[1][0]) - 1 + countExtra(node[2]) + countExtra(node[3])


def objective(m, X, y):
    """R(T) recomputed from the tree alone, to check the cost fit() accumulated"""
    err = errorRate(y, m.predict(X))
    return err + m.lam * countLeaves(m.tree) + m.gamma * countExtra(m.tree)