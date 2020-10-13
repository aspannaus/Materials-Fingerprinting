import numpy as np
import ctypes as ct
import numpy.ctypeslib as npct
from math import pow, fabs
from numpy import power
from numpy.linalg import norm
from scipy.optimize import linear_sum_assignment
# from c_dist import dpc, wd
from numpy.ctypeslib import ndpointer
import os


# setup interface with the dpc c-lib
_doublepp = ndpointer(dtype=np.uintp, ndim=1, flags='C')
_dpc = npct.load_library('./libdpc.so', os.path.dirname(__file__))
# Define the return type of the C function
_dpc.dpc.restype = ct.c_double
# Define arguments of the C function
_dpc.dpc.argtypes = [_doublepp, _doublepp, ct.c_double,
                     ct.c_double, ct.c_int, ct.c_int]


def to_ptr(x):
    xptr = (x.__array_interface__['data'][0]
            + np.arange(x.shape[0])*x.strides[0]).astype(np.uintp)
    return xptr


class PDMetric():
    """ Persistence diagram metric class"""
    def __init__(self, metric):
        if metric == 'wd':
            self.e = [-1, -1, -1]
        elif metric is 'dpc':
            self.e = [1, 0.05, 1.05]

    def dist_fun(self, d1, d2, p, dim):
        n = d1.shape[0]  # rows
        m = d2.shape[0]  # cols
        if n == 0 and m == 0:
            return 0.0
        elif n == m and np.allclose(d1, d2):
            return 0.0

        if self.e[dim] < 0:  # wasserstein
            if n == 0:
                d1 = np.zeros((m, 2))
                n = m
            elif m == 0:
                d2 = np.zeros((n, 2))
                m = n
            size = n + m
            dm = np.zeros((size, size))
            M = np.zeros((size, size))
            return wd(d1, d2, dm, M, p, m, n)
        else:  # dpc
            if n == 0 or m == 0:
                tmp = pow(self.e[dim], p) * max(m, n)
                return pow(tmp, 1.0 / p)
            elif m < n:  # n \leq m by the defn
                n, m = m, n
                d1, d2 = d2, d1
            d1_ptr = to_ptr(d1)
            d2_ptr = to_ptr(d2)
            return _dpc.dpc(d1_ptr, d2_ptr, ct.c_double(p), ct.c_double(eps),
                            ct.c_int(n), ct.c_int(m))


def proj(pt):
    """Compute projection onto y=x diagonal for wd."""
    tmp = (pt[0] + pt[1]) * 0.5
    return tmp


def dpc_py(d1, d2, eps, p):
    """Compute p-th dpc distance between persistence diagrams."""
    n = d1.shape[0]  # rows
    m = d2.shape[0]  # cols
    dm = np.zeros([n, m])

    for i in range(n):
        for j in range(m):
            dm[i, j] = norm((d1[dim][i] - d2[dim][j]), ord=np.inf)
    idx = np.where(dm > eps)
    dm[idx] = eps
    dm = power(dm, p)
    sol = linear_sum_assignment(dm)
    cost = dm[sol[0], sol[1]].sum()
    cost += (m - n) * pow(eps, p)
    cost *= (1 / m)
    return pow(cost, 1 / p)


def dpc_cy(d1, d2, eps, p):
    """
    Wrapper for cython implementation of dpc distance
    """
    n = d1.shape[0]  # rows
    m = d2.shape[0]  # cols

    dm = np.zeros([n, m])

    return dpc(d1, d2, dm, p, eps)


def wd_py(D1, D2, p):
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
