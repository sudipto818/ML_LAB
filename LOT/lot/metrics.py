import numpy as np

# fraction of wrong predictions
def error_rate(y_true, y_pred):
    wrong = 0
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            wrong += 1
    return wrong / len(y_true)


# S(T): number of leaves of a tree 
def count_leaves(node):
    if node[0] == "leaf":
        return 1
    return count_leaves(node[2]) + count_leaves(node[3])


# number of split nodes
def count_splits(node):
    if node[0] == "leaf":
        return 0
    return 1 + count_splits(node[2]) + count_splits(node[3])


# total nonzero coefficients over all tests: an axis test counts 1, a pair 2, a triple 3
def count_nnz(node):
    if node[0] == "leaf":
        return 0
    return len(node[1][0]) + count_nnz(node[2]) + count_nnz(node[3])


# sum of c(a) = ||a||_0 - 1 over all tests: 0 for axis tests
def count_extra(node):
    if node[0] == "leaf":
        return 0
    return len(node[1][0]) - 1 + count_extra(node[2]) + count_extra(node[3])


# R(T) = training error + lam * (number of leaves) + gamma * sum of c(a), recomputed from the tree
# this does not use the cost that fit() accumulated, so it can be used to check it
def objective(m, X, y):
    err = error_rate(y, m.predict(X))
    return err + m.lam * count_leaves(m.tree) + m.gamma * count_extra(m.tree)