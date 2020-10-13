#!python
# cython: language_level=3
import numpy as np
from libc.math cimport pow, abs
from scipy.optimize import linear_sum_assignment
from numpy.linalg import norm
cimport numpy as np
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
np.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
def inf_norm_diff(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] y):
    return max(abs(x[0] - y[0]), abs(x[1] - y[1]))


# turn off bounds-checking for entire function
@cython.boundscheck(False)
# turn off negative index wrapping for entire function
@cython.wraparound(False)
def dpc(np.ndarray[DTYPE_t, ndim=2] d1, np.ndarray[DTYPE_t, ndim=2] d2,
        np.ndarray[DTYPE_t, ndim=2] dm, DTYPE_t p, DTYPE_t eps):
    cdef double cost = 0.0
    cdef double d = 0.0
    cdef int n = d1.shape[0]
    cdef int m = d2.shape[0]

    for i in range(n):
        for j in range(m):
            d = inf_norm_diff(d1[i], d2[j])
            dm[i, j] = pow(min(eps, d), p)
    row_sol, col_sol = linear_sum_assignment(dm)
    for i in range(len(row_sol)):
        cost = cost + dm[row_sol[i]*n+col_sol[i]]
    cost = cost + (m - n) * pow(eps, p)
    cost = (1.0 / m) * cost
    cost = pow(cost, 1.0 / p)
    return cost


def proj(np.ndarray[DTYPE_t, ndim=1] pt):
    cdef double tmp = (pt[0] + pt[1]) * 0.5
    return tmp


@cython.boundscheck(False)
@cython.wraparound(False)
def inf_norm(np.ndarray[DTYPE_t, ndim=1] x):
    return max(abs(x[0]), abs(x[1]))


@cython.boundscheck(False)
@cython.wraparound(False)
def wd(np.ndarray[DTYPE_t, ndim=2] D1, np.ndarray[DTYPE_t, ndim=2] D2,
        np.ndarray[DTYPE_t, ndim=2] dm, np.ndarray[DTYPE_t, ndim=2] M,
        DTYPE_t p, int m, int n):
    cdef double cost = 0.0
    cdef double d = 0.0
    cdef double tmp = 0.0
    cdef int size = n + m

    for i in range(n):
        for j in range(m):
            dm[i, j] = pow(norm(D1[i] - D2[j]), p)
        for j in range(m, size):
            tmp = proj(D1[i])
            dm[i, j] = pow(tmp, p)

    for i in range(n, size):
        for j in range(m):
            tmp = proj(D2[j])
            dm[i, j] = pow(tmp, p)

    row_idxs, col_idxs = linear_sum_assignment(dm)
    M[row_idxs, col_idxs] = 1
    # assume a perfect matching
    for i in range(size):
        for j in range(size):
            if M[i, j] == 1:
                if i >= n:
                    if j < m:
                        tmp = proj(D2[j])
                        cost += pow(tmp, p)
                else:
                    if j >= m:
                        tmp = proj(D1[i])
                        cost += pow(tmp, p)
                    else:
                        cost += pow(norm(D1[i] - D2[j]), p)

    return pow(cost, 1 / p)
