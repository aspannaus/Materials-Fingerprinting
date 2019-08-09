#!/usr/bin/env python3

import numpy as np
import multiprocessing as mp
from multiprocessing.managers import BaseManager
import gc
from distances import dpc, wd


class distances():
    """Compute Persistence diagram metric class."""

    def __init__(self, data, metric):
        """Initialize pd distance class.

        params: data -
        metric: string {'dpc', 'wd'} for either
                dpc or wasserstein distance
        """
        if metric is 'dpc':
            self.e = [1., 0.15, 1.05]
        elif metric is 'wd':
            self.e = [-1, -1, -1]
        else:
            print(" Invalid metric specified. Must be dpc or wd.")
            metric = input(" --> ")
        self.p = 2
        self.var_f = np.zeros((data.fcc_len, 6))
        self.mean_f = np.zeros((data.fcc_len, 6))
        self.var_b = np.zeros((data.bcc_len, 6))
        self.mean_b = np.zeros((data.bcc_len, 6))
        self.tmp = np.zeros(data.bcc_len)
        self.tmp1 = np.zeros(data.fcc_len)
        self.m = np.zeros(data.bcc_len+data.fcc_len)
        self.v = np.zeros(data.bcc_len+data.fcc_len)
        self.X = np.zeros((data.bcc_len+data.fcc_len, 12))
        self.y = np.zeros(data.full_len)
        self.y[:data.bcc_len] = -1  # bcc
        self.y[-data.fcc_len:] = 1  # fcc

    def dist(self, d1, d2, p, eps):
        """Compute persistence diagram metric."""
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
            return wd(d1, d2, p)
        else:  # dpc
            if n == 0 or m == 0:
                return eps
            elif m < n:  # n \leq m by definition
                n, m = m, n
                d1, d2 = d2, d1

            return dpc(d1, d2, p, eps)

    def compute_bcc(self, data, dim):
        """
        Compute pd distance between bcc diagrams of homological dimension dim.
        """
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
        """
        Compute pd distance between fcc diagrams of homological dimension dim.
        """
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

    def feature_matrix(self):
        """Create feature matrix from distances."""
        self.m = np.vstack((self.mean_b, self.mean_f))
        self.s = np.vstack((self.var_b, self.var_f))
        self.X = np.concatenate((self.m, self.s), axis=1)
        return None

    def dists_mp(self, data):
        """Compute pd diances in parallel."""
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
        """
        Compute pd distance between bcc diagrams of homological dimension dim.
        """
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
        print(' BCC distances dim: {:d} complete'.format(dim))

    def fcc_mp(self, data, means, var, ret_dict, dim, level):
        """
        Compute pd distance between fcc diagrams of homological dimension dim.
        """
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
        print(' FCC distances dim: {:d} complete'.format(dim))


class dgmManager(BaseManager):
    pass
