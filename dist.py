#!/usr/bin/env python3

import numpy as np
import classify_utils
import multiprocessing as mp
from multiprocessing.managers import BaseManager
import numpy.ctypeslib as npct
from numpy.ctypeslib import ndpointer
import ctypes as ct
import os
from distances import dpc_cy
from c_dist import wd
import gc
import time

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


class distances():
    def __init__(self, data, metric, multi=False):
        if metric == 'dpc':
            self.e = [1., 0.05, 1.05]
        else:
            self.e = [-1, -1, -1]
        self.p = 2
        if multi is False:
            self.var_f = np.zeros((data.fcc_len, 6))
            self.mean_f = np.zeros((data.fcc_len, 6))
            self.var_b = np.zeros((data.bcc_len, 6))
            self.mean_b = np.zeros((data.bcc_len, 6))
            self.tmp = np.zeros(data.bcc_len)
            self.tmp1 = np.zeros(data.fcc_len)
            self.X = np.zeros((data.bcc_len+data.fcc_len, 12))
            self.y = np.zeros(data.full_len)
            self.y[:data.bcc_len] = -1  # bcc
            self.y[-data.fcc_len:] = 1  # fcc
        else:
            self.var_f = np.zeros((data.full_len, 3))
            self.mean_f = np.zeros((data.full_len, 3))
            self.var_b = np.zeros((data.full_len, 3))
            self.mean_b = np.zeros((data.full_len, 3))
            self.tmp = np.zeros(data.bcc_len)
            self.tmp1 = np.zeros(data.fcc_len)
            self.X = np.zeros((data.full_len, 12))
            in_file = '~/GitHub/fingerprint/multiphase/multiphase_y.txt'
            self.y = np.loadtxt(in_file, np.float, max_rows=data.full_len)

    def dist(self, d1, d2, p, eps):
        n = d1.shape[0]  # rows
        m = d2.shape[0]  # cols
        if n == 0 and m == 0:
            return 0.0
        elif n == m and np.allclose(d1, d2):
            return 0.0

        if eps < 0:  # wasserstein
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
                return eps
            elif m < n:  # n \leq m by the defn
                n, m = m, n
                d1, d2 = d2, d1

            d1_ptr = to_ptr(d1)
            d2_ptr = to_ptr(d2)
            return _dpc.dpc(d1_ptr, d2_ptr, ct.c_double(p), ct.c_double(eps),
                            ct.c_int(n), ct.c_int(m))

    def compute_bcc(self, data, dim):
        print('\n Computing BCC distances dim: {:d}'.format(dim))
        for i, dgm in enumerate(data.bcc_dgms):
            for j, dgm1 in enumerate(data.bcc_dgms):
                self.tmp[j] = self.dist(dgm[dim], dgm1[dim],
                                        self.p, self.e[dim])
            for j, dgm2 in enumerate(data.fcc_dgms):
                self.tmp1[j] = self.dist(dgm[dim], dgm2[dim],
                                         self.p, self.e[dim])
            self.mean_b[i, dim] = np.mean(self.tmp)
            self.mean_b[i, dim+3] = np.mean(self.tmp1)
            self.var_b[i, dim] = np.var(self.tmp, ddof=1)
            self.var_b[i, dim+3] = np.var(self.tmp1, ddof=1)

    def compute_fcc(self, data, dim):
        print('\n Computing FCC distances dim: {:d}'.format(dim))
        for i, dgm in enumerate(data.fcc_dgms):
            for j, dgm1 in enumerate(data.bcc_dgms):
                self.tmp[j] = self.dist(dgm[dim], dgm1[dim],
                                        self.p, self.e[dim])
            for j, dgm2 in enumerate(data.fcc_dgms):
                self.tmp1[j] = self.dist(dgm[dim], dgm2[dim],
                                         self.p, self.e[dim])
            self.mean_f[i, dim] = np.mean(self.tmp)
            self.var_f[i, dim] = np.var(self.tmp, ddof=1)
            self.mean_f[i, dim+3] = np.mean(self.tmp1)
            self.var_f[i, dim+3] = np.var(self.tmp1, ddof=1)

    def feature_matrix(self, data, multiphase=False):
        if multiphase:
            self.m = np.hstack((self.mean_b, self.mean_f))
            self.s = np.hstack((self.var_b, self.var_f))
            self.X = np.concatenate((self.m, self.s), axis=1)
            # print(self.X)
        else:
            self.m = np.vstack((self.mean_b, self.mean_f))
            self.s = np.vstack((self.var_b, self.var_f))
            self.X = np.concatenate((self.m, self.s), axis=1)
        return None

    def dists_mp(self, data):
        m0_b = np.zeros((data.bcc_len, 2))
        m1_b = np.zeros((data.bcc_len, 2))
        m2_b = np.zeros((data.bcc_len, 2))
        v0_b = np.zeros((data.bcc_len, 2))
        v1_b = np.zeros((data.bcc_len, 2))
        v2_b = np.zeros((data.bcc_len, 2))
        m0_f = np.zeros((data.fcc_len, 2))
        m1_f = np.zeros((data.fcc_len, 2))
        m2_f = np.zeros((data.fcc_len, 2))
        v0_f = np.zeros((data.fcc_len, 2))
        v1_f = np.zeros((data.fcc_len, 2))
        v2_f = np.zeros((data.fcc_len, 2))
        manager = mp.Manager()
        ret_dict = manager.dict()

        p1 = mp.Process(target=self.bcc_mp, args=(data, m0_b, v0_b,
                                                  ret_dict, 0, 'b0'))
        p2 = mp.Process(target=self.bcc_mp, args=(data, m1_b, v1_b,
                                                  ret_dict, 1, 'b1'))
        p3 = mp.Process(target=self.fcc_mp, args=(data, m0_f, v0_f,
                                                  ret_dict, 0, 'f0'))
        p4 = mp.Process(target=self.fcc_mp, args=(data, m1_f, v1_f,
                                                  ret_dict, 1, 'f1'))
        p5 = mp.Process(target=self.bcc_mp, args=(data, m2_b, v2_b,
                                                  ret_dict, 2, 'b2'))
        p6 = mp.Process(target=self.fcc_mp, args=(data, m2_f, v2_f,
                                                  ret_dict, 2, 'f2'))
        print(' Computing all distances')
        p1.start()
        p2.start()
        p3.start()
        p4.start()
        p5.start()
        p6.start()

        p1.join()
        p2.join()
        p3.join()
        p4.join()
        p5.join()
        p6.join()
        print(' Done')

        for k in range(3):
            b_level = 'b' + str(k)
            f_level = 'f' + str(k)
            self.mean_b[:, k] = ret_dict[b_level][0][:, 0].copy()
            self.mean_b[:, k+3] = ret_dict[b_level][0][:, 1].copy()
            self.var_b[:, k] = ret_dict[b_level][1][:, 0].copy()
            self.var_b[:, k+3] = ret_dict[b_level][1][:, 1].copy()
            self.mean_f[:, k] = ret_dict[f_level][0][:, 0].copy()
            self.mean_f[:, k+3] = ret_dict[f_level][0][:, 1].copy()
            self.var_f[:, k] = ret_dict[f_level][1][:, 0].copy()
            self.var_f[:, k+3] = ret_dict[f_level][1][:, 1].copy()

    def bcc_mp(self, data, means, var, ret_dict, dim, level):
        t = time.time()
        print(' Computing BCC distances dim: {:d}'.format(dim))
        for i, dgm in enumerate(data.bcc_dgms):
            for j, dgm1 in enumerate(data.bcc_dgms):
                self.tmp[j] = self.dist(dgm[dim], dgm1[dim],
                                        self.p, self.e[dim])
            for j, dgm2 in enumerate(data.fcc_dgms):
                self.tmp1[j] = self.dist(dgm[dim], dgm2[dim],
                                         self.p, self.e[dim])
            means[i, 0] = np.mean(self.tmp)
            var[i, 0] = np.var(self.tmp, ddof=1)
            means[i, 1] = np.mean(self.tmp1)
            var[i, 1] = np.var(self.tmp1, ddof=1)
            gc.collect()
        ret_dict[level] = [means, var]
        print(' BCC distances dim: {:d} BxF={}x{} completed in {:15.4f} seconds'.format(dim,len(data.bcc_dgms),len(data.fcc_dgms),time.time()-t))

    def fcc_mp(self, data, means, var, ret_dict, dim, level):
        t = time.time()
        print(' Computing FCC distances dim: {:d}'.format(dim))
        for i, dgm in enumerate(data.fcc_dgms):
            for j, dgm1 in enumerate(data.bcc_dgms):
                self.tmp[j] = self.dist(dgm[dim], dgm1[dim],
                                        self.p, self.e[dim])
            for j, dgm2 in enumerate(data.fcc_dgms):
                self.tmp1[j] = self.dist(dgm[dim], dgm2[dim],
                                         self.p, self.e[dim])
            means[i, 0] = np.mean(self.tmp)
            var[i, 0] = np.var(self.tmp, ddof=1)
            means[i, 1] = np.mean(self.tmp1)
            var[i, 1] = np.var(self.tmp1, ddof=1)
            gc.collect()
        ret_dict[level] = [means, var]
        print(' FCC distances dim: {:d} FxB={}x{} completed in {:15.4f} seconds'.format(dim,len(data.fcc_dgms),len(data.bcc_dgms),time.time()-t))

    def dists_mp_multiphase(self, data, train):
        m0_b = np.zeros((data.full_len, 1))
        m1_b = np.zeros((data.full_len, 1))
        m2_b = np.zeros((data.full_len, 1))
        v0_b = np.zeros((data.full_len, 1))
        v1_b = np.zeros((data.full_len, 1))
        v2_b = np.zeros((data.full_len, 1))
        m0_f = np.zeros((data.full_len, 1))
        m1_f = np.zeros((data.full_len, 1))
        m2_f = np.zeros((data.full_len, 1))
        v0_f = np.zeros((data.full_len, 1))
        v1_f = np.zeros((data.full_len, 1))
        v2_f = np.zeros((data.full_len, 1))
        manager = mp.Manager()
        ret_dict = manager.dict()

        p1 = mp.Process(target=self.bcc_mp_multiphase,
                        args=(data, train, m0_b, v0_b, ret_dict, 0, 'b0'))
        p2 = mp.Process(target=self.bcc_mp_multiphase,
                        args=(data, train, m1_b, v1_b, ret_dict, 1, 'b1'))
        p3 = mp.Process(target=self.fcc_mp_multiphase,
                        args=(data, train, m0_f, v0_f, ret_dict, 0, 'f0'))
        p4 = mp.Process(target=self.fcc_mp_multiphase,
                        args=(data, train, m1_f, v1_f, ret_dict, 1, 'f1'))
        p5 = mp.Process(target=self.bcc_mp_multiphase,
                        args=(data, train, m2_b, v2_b, ret_dict, 2, 'b2'))
        p6 = mp.Process(target=self.fcc_mp_multiphase,
                        args=(data, train, m2_f, v2_f, ret_dict, 2, 'f2'))
        print(' Computing all multiphase distances')
        p1.start()
        p2.start()
        p3.start()
        p4.start()
        p5.start()
        p6.start()

        p1.join()
        p2.join()
        p3.join()
        p4.join()
        p5.join()
        p6.join()
        print(' Done')

        for k in range(3):
            b_level = 'b' + str(k)
            f_level = 'f' + str(k)
            self.mean_b[:, k] = ret_dict[b_level][0][:, 0].copy()
            self.var_b[:, k] = ret_dict[b_level][1][:, 0].copy()
            self.mean_f[:, k] = ret_dict[f_level][0][:, 0].copy()
            self.var_f[:, k] = ret_dict[f_level][1][:, 0].copy()

    def bcc_mp_multiphase(self, data, train, means, var, ret_dict, dim, level):
        t = time.time()
        print(' Computing BCC distances dim: {:d}'.format(dim))
        for i, dgm in enumerate(data.dgms):
            for j, dgm1 in enumerate(train.bcc_dgms):
                self.tmp[j] = self.dist(dgm[dim], dgm1[dim],
                                        self.p, self.e[dim])
            means[i, 0] = np.mean(self.tmp)
            var[i, 0] = np.var(self.tmp, ddof=1)
            gc.collect()
        ret_dict[level] = [means, var]
        print(' BCC distances dim: {:d} MxB={}x{} completed in {:15.4f} seconds'.format(dim,len(data.dgms),len(train.bcc_dgms),time.time()-t))

    def fcc_mp_multiphase(self, data, train, means, var, ret_dict, dim, level):
        t = time.time()
        print(' Computing FCC distances dim: {:d}'.format(dim))
        for i, dgm in enumerate(data.dgms):
            for j, dgm1 in enumerate(train.fcc_dgms):
                self.tmp1[j] = self.dist(dgm[dim], dgm1[dim],
                                         self.p, self.e[dim])
            means[i, 0] = np.mean(self.tmp1)
            var[i, 0] = np.var(self.tmp1, ddof=1)
            gc.collect()
        ret_dict[level] = [means, var]
        print(' FCC distances dim: {:d} MxF={}x{} completed in {:15.4f} seconds'.format(dim,len(data.dgms),len(train.fcc_dgms),time.time()-t))


class dgmManager(BaseManager):
    pass
