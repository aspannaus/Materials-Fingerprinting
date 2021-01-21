#!/usr/bin/env python3

import numpy as np
import multiprocessing as mp
from multiprocessing.managers import BaseManager
from distances import dpc, wd
import gc


class distances():
    def __init__(self, data, metric, classes=['bcc', 'fcc']):
        if metric == 'dpc':
            self.e = [1., 0.05, 1.05]
        else:
            self.e = [-1, -1, -1]
        self.p = 2
        classes.sort()
        self.classes = classes
        self.num_classes = len(self.classes)
        self.y = np.zeros(data.full_len)
        if self.num_classes == 2:
            self.n = 6
            self.P = 12
            if classes[0] == 'bcc':  # bcc dgms to compare
                self.len1 = data.bcc_len
                self.type1 = data.bcc_dgms
                self.type1_name = self.classes[0]
            elif classes[0] == 'fcc':
                self.len1 = data.fcc_len
                self.type1 = data.fcc_dgms
                self.type1_name = self.classes[0]
            if classes[1] == 'fcc':
                self.len2 = data.fcc_len
                self.type2 = data.fcc_dgms
                self.type2_name = self.classes[1]
            else:
                self.len2 = data.hcp_len
                self.type2 = data.hcp_dgms
                self.type2_name = self.classes[1]
            self.y[:self.len1] = 0  # type 1, classes[0]
            self.y[-self.len2:] = 1  # type 2, classes[1]
        else:  # multi-class!
            self.type1_name = 'BCC'
            self.type2_name = 'FCC'
            self.type3_name = 'HCP'
            self.type1 = data.bcc_dgms
            self.type2 = data.fcc_dgms
            self.type3 = data.hcp_dgms
            self.len1 = data.bcc_len
            self.len2 = data.fcc_len
            self.len3 = data.hcp_len
            self.n = 9
            self.P = 18
            self.var_3 = np.zeros((self.len3, self.n))
            self.mean_3 = np.zeros((self.len3, self.n))
            self.tmp3 = np.zeros(self.len3)
            self.y[:self.len1] = 0  # type 1, classes[0]
            self.y[self.len1:-self.len2] = 1  # type 2, classes[1]
            self.y[-self.len2:] = 2  # type 2, classes[2]

        self.var_2 = np.zeros((self.len2, self.n))
        self.mean_2 = np.zeros((self.len2, self.n))
        self.var_1 = np.zeros((self.len1, self.n))
        self.mean_1 = np.zeros((self.len1, self.n))
        self.tmp1 = np.zeros(self.len1)
        self.tmp2 = np.zeros(self.len2)
        self.m = np.zeros(data.full_len)
        self.v = np.zeros(data.full_len)
        self.X = np.zeros([data.full_len, self.P])

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
            return wd(d1, d2, p)
        else:  # dpc
            if n == 0 or m == 0:
                return eps
            elif m < n:  # n \leq m by the defn
                n, m = m, n
                d1, d2 = d2, d1

            return dpc(d1, d2, p, eps)

    def compute_type1(self, dim):
        print('\n Computing {:s} distances dim: {:d}'.format(self.type1_name, dim))
        if self.num_classes == 2:
            for i, dgm in enumerate(self.type1):
                for j, dgm1 in enumerate(self.type1):
                    self.tmp1[j] = self.dist(dgm[dim], dgm1[dim],
                                             self.p, self.e[dim])
                for j, dgm2 in enumerate(self.type2):
                    self.tmp2[j] = self.dist(dgm[dim], dgm2[dim],
                                             self.p, self.e[dim])
                self.mean_1[i, dim] = np.mean(self.tmp1)
                self.mean_1[i, dim+3] = np.mean(self.tmp2)
                self.var_1[i, dim] = np.var(self.tmp1, ddof=1)
                self.var_1[i, dim+3] = np.var(self.tmp2, ddof=1)
                gc.collect()
        else:
            for i, dgm in enumerate(self.type1):
                for j, dgm1 in enumerate(self.type1):
                    self.tmp1[j] = self.dist(dgm[dim], dgm1[dim],
                                             self.p, self.e[dim])
                for j, dgm2 in enumerate(self.type2):
                    self.tmp2[j] = self.dist(dgm[dim], dgm2[dim],
                                             self.p, self.e[dim])
                for j, dgm3 in enumerate(self.type3):
                    self.tmp3[j] = self.dist(dgm[dim], dgm3[dim],
                                             self.p, self.e[dim])
                self.mean_1[i, dim] = np.mean(self.tmp1)
                self.var_1[i, dim] = np.var(self.tmp1, ddof=1)
                self.mean_1[i, dim+3] = np.mean(self.tmp2)
                self.var_1[i, dim+3] = np.var(self.tmp2, ddof=1)
                self.mean_1[i, dim+6] = np.mean(self.tmp3)
                self.var_1[i, dim+6] = np.var(self.tmp3, ddof=1)
                gc.collect()
        print(' Distances {:s} dim: {:d} Complete!'.format(self.type1_name, dim))

    def compute_type2(self, dim):
        print('\n Computing {:s} distances dim: {:d}'.format(self.type2_name, dim))
        if self.num_classes == 2:
            for i, dgm in enumerate(self.type2):
                for j, dgm1 in enumerate(self.type1):
                    self.tmp1[j] = self.dist(dgm[dim], dgm1[dim],
                                             self.p, self.e[dim])
                for j, dgm2 in enumerate(self.type2):
                    self.tmp2[j] = self.dist(dgm[dim], dgm2[dim],
                                             self.p, self.e[dim])
                self.mean_2[i, dim] = np.mean(self.tmp1)
                self.var_2[i, dim] = np.var(self.tmp1, ddof=1)
                self.mean_2[i, dim+3] = np.mean(self.tmp2)
                self.var_2[i, dim+3] = np.var(self.tmp2, ddof=1)
                gc.collect()
        else:
            for i, dgm in enumerate(self.type2):
                for j, dgm1 in enumerate(self.type1):
                    self.tmp1[j] = self.dist(dgm[dim], dgm1[dim],
                                             self.p, self.e[dim])
                for j, dgm2 in enumerate(self.type2):
                    self.tmp2[j] = self.dist(dgm[dim], dgm2[dim],
                                             self.p, self.e[dim])
                for j, dgm3 in enumerate(self.type3):
                    self.tmp3[j] = self.dist(dgm[dim], dgm3[dim],
                                             self.p, self.e[dim])
                self.mean_2[i, dim] = np.mean(self.tmp1)
                self.var_2[i, dim] = np.var(self.tmp1, ddof=1)
                self.mean_2[i, dim+3] = np.mean(self.tmp2)
                self.var_2[i, dim+3] = np.var(self.tmp2, ddof=1)
                self.mean_2[i, dim+6] = np.mean(self.tmp3)
                self.var_2[i, dim+6] = np.var(self.tmp3, ddof=1)
                gc.collect()
        print(' Distances {:s} dim: {:d} Complete!'.format(self.type2_name, dim))

    def compute_type3(self, dim):
        print('\n Computing {:s} distances dim: {:d}'.format(self.type3_name, dim))
        for i, dgm in enumerate(self.type3):
            for j, dgm1 in enumerate(self.type1):
                self.tmp1[j] = self.dist(dgm[dim], dgm1[dim],
                                         self.p, self.e[dim])
            for j, dgm2 in enumerate(self.type2):
                self.tmp2[j] = self.dist(dgm[dim], dgm2[dim],
                                         self.p, self.e[dim])
            for j, dgm3 in enumerate(self.type3):
                self.tmp3[j] = self.dist(dgm[dim], dgm3[dim],
                                         self.p, self.e[dim])
            self.mean_3[i, dim] = np.mean(self.tmp1)
            self.var_3[i, dim] = np.var(self.tmp1, ddof=1)
            self.mean_3[i, dim+3] = np.mean(self.tmp2)
            self.var_3[i, dim+3] = np.var(self.tmp2, ddof=1)
            self.mean_3[i, dim+6] = np.mean(self.tmp3)
            self.var_3[i, dim+6] = np.var(self.tmp3, ddof=1)
            gc.collect()
        print(' Distances {:s} dim: {:d} Complete!'.format(self.type3_name, dim))

    def feature_matrix(self):
        """Concatenate summary statistic vectors for feature matrix."""
        if self.num_classes == 2:
            self.m = np.vstack((self.mean_1, self.mean_2))
            self.s = np.vstack((self.var_1, self.var_2))
        else:
            self.m = np.vstack((self.mean_1, self.mean_2, self.mean_3))
            self.s = np.vstack((self.var_1, self.var_2, self.var_3))
        self.X = np.concatenate((self.m, self.s), axis=1)

        return None

    def dists_mp(self, data):
        manager = mp.Manager()
        ret_dict = manager.dict()
        m0_t1 = np.zeros((self.len1, self.num_classes))
        m1_t1 = np.zeros((self.len1, self.num_classes))
        m2_t1 = np.zeros((self.len1, self.num_classes))
        v0_t1 = np.zeros((self.len1, self.num_classes))
        v1_t1 = np.zeros((self.len1, self.num_classes))
        v2_t1 = np.zeros((self.len1, self.num_classes))
        m0_t2 = np.zeros((self.len2, self.num_classes))
        m1_t2 = np.zeros((self.len2, self.num_classes))
        m2_t2 = np.zeros((self.len2, self.num_classes))
        v0_t2 = np.zeros((self.len2, self.num_classes))
        v1_t2 = np.zeros((self.len2, self.num_classes))
        v2_t2 = np.zeros((self.len2, self.num_classes))
        if self.num_classes == 3:
            m0_t3 = np.zeros((self.len3, self.num_classes))
            m1_t3 = np.zeros((self.len3, self.num_classes))
            m2_t3 = np.zeros((self.len3, self.num_classes))
            v0_t3 = np.zeros((self.len3, self.num_classes))
            v1_t3 = np.zeros((self.len3, self.num_classes))
            v2_t3 = np.zeros((self.len3, self.num_classes))
            p7 = mp.Process(target=self.type3_mp, args=(m0_t3, v0_t3,
                                                        ret_dict, 0, 't3_0'))
            p8 = mp.Process(target=self.type3_mp, args=(m1_t3, v1_t3,
                                                        ret_dict, 1, 't3_1'))
            p9 = mp.Process(target=self.type3_mp, args=(m2_t3, v2_t3,
                                                        ret_dict, 2, 't3_2'))

        p1 = mp.Process(target=self.type1_mp, args=(m0_t1, v0_t1,
                                                    ret_dict, 0, 't1_0'))
        p2 = mp.Process(target=self.type1_mp, args=(m1_t1, v1_t1,
                                                    ret_dict, 1, 't1_1'))
        p3 = mp.Process(target=self.type1_mp, args=(m2_t1, v2_t1,
                                                    ret_dict, 2, 't1_2'))
        p4 = mp.Process(target=self.type2_mp, args=(m0_t2, v0_t2,
                                                    ret_dict, 0, 't2_0'))
        p5 = mp.Process(target=self.type2_mp, args=(m1_t2, v1_t2,
                                                    ret_dict, 1, 't2_1'))
        p6 = mp.Process(target=self.type2_mp, args=(m2_t2, v2_t2,
                                                    ret_dict, 2, 't2_2'))

        print(' Computing all distances')
        p1.start()
        p2.start()
        p3.start()
        p4.start()
        p5.start()
        p6.start()
        if self.num_classes == 3:
            p7.start()
            p8.start()
            p9.start()

        p1.join()
        p2.join()
        p3.join()
        p4.join()
        p5.join()
        p6.join()
        if self.num_classes == 3:
            p7.join()
            p8.join()
            p9.join()
        print(' Done')

        for k in range(3):
            t1_level = 't1_' + str(k)
            t2_level = 't2_' + str(k)
            self.mean_1[:, k] = ret_dict[t1_level][0][:, 0].copy()
            self.mean_1[:, k+3] = ret_dict[t1_level][0][:, 1].copy()
            self.var_1[:, k] = ret_dict[t1_level][1][:, 0].copy()
            self.var_1[:, k+3] = ret_dict[t1_level][1][:, 1].copy()
            self.mean_2[:, k] = ret_dict[t2_level][0][:, 0].copy()
            self.mean_2[:, k+3] = ret_dict[t2_level][0][:, 1].copy()
            self.var_2[:, k] = ret_dict[t2_level][1][:, 0].copy()
            self.var_2[:, k+3] = ret_dict[t2_level][1][:, 1].copy()
            if self.num_classes == 3:
                t3_level = 't3_' + str(k)
                self.mean_1[:, k+6] = ret_dict[t1_level][0][:, 2].copy()
                self.var_1[:, k+6] = ret_dict[t1_level][1][:, 2].copy()
                self.mean_2[:, k+6] = ret_dict[t2_level][0][:, 2].copy()
                self.var_2[:, k+6] = ret_dict[t2_level][1][:, 2].copy()
                self.mean_3[:, k] = ret_dict[t3_level][0][:, 0].copy()
                self.mean_3[:, k+3] = ret_dict[t3_level][0][:, 1].copy()
                self.mean_3[:, k+6] = ret_dict[t3_level][0][:, 2].copy()
                self.var_3[:, k] = ret_dict[t3_level][1][:, 0].copy()
                self.var_3[:, k+3] = ret_dict[t3_level][1][:, 1].copy()
                self.var_3[:, k+6] = ret_dict[t3_level][1][:, 2].copy()

    def type1_mp(self, means, var, ret_dict, dim, level):
        """
        Compute pd distance between bcc diagrams of homological dimension dim.
        """
        print(' Computing {:s} distances dim: {:d}'.format(self.type1_name, dim))
        if self.num_classes == 2:
            for i, dgm in enumerate(self.type1):
                for j, dgm1 in enumerate(self.type1):
                    self.tmp1[j] = self.dist(dgm[dim], dgm1[dim],
                                             self.p, self.e[dim])
                for j, dgm2 in enumerate(self.type2):
                    self.tmp2[j] = self.dist(dgm[dim], dgm2[dim],
                                             self.p, self.e[dim])
                means[i, 0] = np.mean(self.tmp1)
                var[i, 0] = np.var(self.tmp1, ddof=1)
                means[i, 1] = np.mean(self.tmp2)
                var[i, 1] = np.var(self.tmp2, ddof=1)
                gc.collect()
        else:  # multiclass!
            for i, dgm in enumerate(self.type1):
                for j, dgm1 in enumerate(self.type1):
                    self.tmp1[j] = self.dist(dgm[dim], dgm1[dim],
                                             self.p, self.e[dim])
                for j, dgm2 in enumerate(self.type2):
                    self.tmp2[j] = self.dist(dgm[dim], dgm2[dim],
                                             self.p, self.e[dim])
                for j, dgm3 in enumerate(self.type3):
                    self.tmp3[j] = self.dist(dgm[dim], dgm3[dim],
                                             self.p, self.e[dim])
                means[i, 0] = np.mean(self.tmp1)
                var[i, 0] = np.var(self.tmp1, ddof=1)
                means[i, 1] = np.mean(self.tmp2)
                var[i, 1] = np.var(self.tmp2, ddof=1)
                means[i, 2] = np.mean(self.tmp3)
                var[i, 2] = np.var(self.tmp3, ddof=1)
                gc.collect()
        ret_dict[level] = [means, var]
        print(' {:s} distances dim: {:d} complete'.format(self.type1_name, dim))

    def type2_mp(self, means, var, ret_dict, dim, level):
        """
        Compute pd distance between fcc diagrams of homological dimension dim.
        """
        print(' Computing {:s} distances dim: {:d}'.format(self.type2_name, dim))
        if self.num_classes == 2:
            for i, dgm in enumerate(self.type2):
                for j, dgm1 in enumerate(self.type1):
                    self.tmp1[j] = self.dist(dgm[dim], dgm1[dim],
                                             self.p, self.e[dim])
                for j, dgm2 in enumerate(self.type2):
                    self.tmp2[j] = self.dist(dgm[dim], dgm2[dim],
                                             self.p, self.e[dim])
                means[i, 0] = np.mean(self.tmp1)
                var[i, 0] = np.var(self.tmp1, ddof=1)
                means[i, 1] = np.mean(self.tmp2)
                var[i, 1] = np.var(self.tmp2, ddof=1)
                gc.collect()
        else:  # multiclass!
            for i, dgm in enumerate(self.type2):
                for j, dgm1 in enumerate(self.type1):
                    self.tmp1[j] = self.dist(dgm[dim], dgm1[dim],
                                             self.p, self.e[dim])
                for j, dgm2 in enumerate(self.type2):
                    self.tmp2[j] = self.dist(dgm[dim], dgm2[dim],
                                             self.p, self.e[dim])
                for j, dgm3 in enumerate(self.type3):
                    self.tmp3[j] = self.dist(dgm[dim], dgm3[dim],
                                             self.p, self.e[dim])
                means[i, 0] = np.mean(self.tmp1)
                var[i, 0] = np.var(self.tmp1, ddof=1)
                means[i, 1] = np.mean(self.tmp2)
                var[i, 1] = np.var(self.tmp2, ddof=1)
                means[i, 2] = np.mean(self.tmp3)
                var[i, 2] = np.var(self.tmp3, ddof=1)
                gc.collect()
        ret_dict[level] = [means, var]
        print(' {:s} distances dim: {:d} complete'.format(self.type2_name, dim))

    def type3_mp(self, means, var, ret_dict, dim, level):
        """
        Compute pd distance between fcc diagrams of homological dimension dim.
        """
        print(' Computing {:s} distances dim: {:d}'.format(self.type3_name, dim))
        for i, dgm in enumerate(self.type3):
            for j, dgm1 in enumerate(self.type1):
                self.tmp1[j] = self.dist(dgm[dim], dgm1[dim],
                                         self.p, self.e[dim])
            for j, dgm2 in enumerate(self.type2):
                self.tmp2[j] = self.dist(dgm[dim], dgm2[dim],
                                         self.p, self.e[dim])
            for j, dgm3 in enumerate(self.type3):
                self.tmp3[j] = self.dist(dgm[dim], dgm3[dim],
                                         self.p, self.e[dim])
            means[i, 0] = np.mean(self.tmp1)
            var[i, 0] = np.var(self.tmp1, ddof=1)
            means[i, 1] = np.mean(self.tmp2)
            var[i, 1] = np.var(self.tmp2, ddof=1)
            means[i, 2] = np.mean(self.tmp3)
            var[i, 2] = np.var(self.tmp3, ddof=1)
            gc.collect()
        ret_dict[level] = [means, var]
        print(' {:s} distances dim: {:d} complete'.format(self.type3_name, dim))


class dgmManager(BaseManager):
    pass
