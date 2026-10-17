import numpy as np

# fraction of wrong predictions
def errorRate(yTrue, yPred):
    wrong = 0
    for i in range(len(yTrue)):
        if yTrue[i] != yPred[i]:
            wrong += 1
    return wrong / len(yTrue)


# S(T): number of leaves of a tree 
def countLeaves(node):
    if node[0] == "leaf":
        return 1
    return countLeaves(node[2]) + countLeaves(node[3])


# number of split nodes
def countSplits(node):
    if node[0] == "leaf":
        return 0
    return 1 + countSplits(node[2]) + countSplits(node[3])


# total nonzero coefficients over all tests: an axis test counts 1, a pair 2, a triple 3
def countNnz(node):
    if node[0] == "leaf":
        return 0
    return len(node[1][0]) + countNnz(node[2]) + countNnz(node[3])


# sum of c(a) = ||a||_0 - 1 over all tests: 0 for axis tests
def countExtra(node):
    if node[0] == "leaf":
        return 0
    return len(node[1][0]) - 1 + countExtra(node[2]) + countExtra(node[3])


# R(T) = training error + lam * (number of leaves) + gamma * sum of c(a), recomputed from the tree
# this does not use the cost that fit() accumulated, so it can be used to check it
def objective(m, X, y):
    err = errorRate(y, m.predict(X))
    return err + m.lam * countLeaves(m.tree) + m.gamma * countExtra(m.tree)