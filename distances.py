import numpy as np
from math import pow, fabs
from numpy import power
from numpy.linalg import norm
from scipy.optimize import linear_sum_assignment


def inf_norm(x, y):
    """Compute inf norm between entries in a pd."""
    return max(fabs(x[0] - y[0]), abs(x[1] - y[1]))


def proj(pt):
    """Compute projection onto y=x diagonal for wd."""
    tmp = (pt[0] + pt[1]) * 0.5
    return tmp


def dpc(d1, d2, p, eps):
    """Compute p-th dpc distance between persistence diagrams."""
    n = d1.shape[0]  # rows
    m = d2.shape[0]  # cols
    dm = np.zeros([n, m])

    for i in range(n):
        for j in range(m):
            dm[i, j] = inf_norm(d1[i], d2[j])
    idx = np.where(dm > eps)
    dm[idx] = eps
    dm = power(dm, p)
    matches = linear_sum_assignment(dm)
    cost = dm[matches[0], matches[1]].sum()
    cost += (m - n) * pow(eps, p)
    cost *= (1.0 / m)
    return pow(cost, 1.0 / p)


def wd(D1, D2, p):
    """Compute p-th Wasserstein distance between persistence diagrams."""
    n = D1.shape[0]  # rows
    m = D2.shape[0]  # cols

    cost = 0.0
    size = n + m
    dm = np.zeros([size, size])
    M = np.zeros([size, size])

    for i in range(n):
        for j in range(m):
            dm[i, j] = pow(norm(D1[i] - D2[j]), p)
            for j in range(m, size):
                dm[i, j] = pow(proj(D1[i]), p)

    for i in range(n, size):
        for j in range(m):
            dm[i, j] = pow(proj(D2[j]), p)

    row_idxs, col_idxs = linear_sum_assignment(dm)
    M[row_idxs, col_idxs] = 1
    # assume a perfect matching
    for i in range(size):
        for j in range(size):
            if M[i, j] == 1:
                if i >= n:
                    if j < m:
                        cost += pow(proj(D2[j]), p)
                else:
                    if j >= m:
                        cost += pow(proj(D1[i]), p)
                    else:
                        cost += pow(norm(D1[i] - D2[j]), p)

    return pow(cost, 1 / p)
