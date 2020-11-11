#!/usr/bin/env python3

import numpy as np
from ripser import ripser
from os.path import exists
from functools import wraps
import sys
import gc
import timeit
import datetime
import time


def timestamp(f=None):
    t = time.time()
    if f is not None:
        f.write(time.ctime(t))
    else:
        str = ' ' + time.ctime()
        print(str)
    return None


def MeasureTime(f):
    @wraps(f)
    def _wrapper(*args, **kwargs):
        gcold = gc.isenabled()
        gc.disable()
        start_time = timeit.default_timer()
        try:
            result = f(*args, **kwargs)
        finally:
            tmp = timeit.default_timer() - start_time
            elapsed = str(datetime.timedelta(seconds=tmp))
            if gcold:
                gc.enable()
            print('\n Wall clock time "{}": {}s'.format(f.__name__, elapsed))
        return result
    return _wrapper


class MeasureBlockTime:
    def __init__(self, name="(block)", no_print=False, disable_gc=True):
        self.name = name
        self.no_print = no_print
        self.disable_gc = disable_gc

    def __enter__(self):
        if self.disable_gc:
            self.gcold = gc.isenabled()
            gc.disable()
        self.start_time = timeit.default_timer()

    def __exit__(self, ty, val, tb):
        self.elapsed = timeit.default_timer() - self.start_time
        if self.disable_gc and self.gcold:
            gc.enable()
        if not self.no_print:
            print('\n Wall clock time "{}": {}s'.format(self.name,
                                                        self.elapsed))
        return False  # re-raise any exceptions


def progress_bar(value, endvalue, bar_length=25):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write('\r Percent Complete : [{0}] {1}%'.
                     format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()
    return None


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


class getData():
    def __init__(self):
        pass

    def get_file(self, in_file):
        while not(exists(in_file)):
            print(" File {:s} does not exist. Enter a valid filename.\n".
                  format(in_file))
            in_file = input("--> ")
            if in_file[0] == 'b':
                self.bcc_file = in_file
                print(self.bcc_file)
            elif in_file[0] == 'f':
                self.fcc_file = in_file
        return None

    def read_data(self, in_file, dgms, multi=False):
        if not multi:
            path = '~/GitHub/fingerprint/data/synthetic_data/bcc_025_05.xyz'
            self.bcc_file = path  # + in_file
            self.get_file(self.bcc_file)
            path = '~/GitHub/fingerprint/data/synthetic_data/fcc_025_05.xyz'
            self.fcc_file = path  # + in_file
            self.get_file(self.fcc_file)
            with open(self.bcc_file) as f_in:
                print(" Reading data from {:s}".format(self.bcc_file))
                dgms.bcc_cells = self.read_file(f_in, dgms.bcc_len)
            with open(self.fcc_file) as f_in:
                print(" Reading data from {:s}".format(self.fcc_file))
                dgms.fcc_cells = self.read_file(f_in, dgms.fcc_len)
        else:
            path = '~/GitHub/fingerprint/multiphase/'
            with open(path + in_file) as f_in:
                print(" Reading data from {}".format(path + in_file))
                dgms.cells = self.read_file(f_in, dgms.full_len)

        return None

    def read_file(self, f, n_config):
        """Read xyz file for point set registration.

        f - open file for reading in xyz format
        returns:
        atom_pts - n x 4 list of atoms and placement
        of the form: atom (as int! placement in .xyz file) x y z
        !! Not implemented yet !!!
        atoms - dictionary with elements as keys
        and each value a list with the elements position in
        the .xyz file.
        """
        atoms = []

        a = f.readline().split()  # number of atoms in configuration
        atom_ct = int(a[0])
        nbrs = np.empty([atom_ct, 3])
        atom_pts = [None] * n_config
        tmp = np.empty(3)
        f.readline()

        for n in range(n_config):
            for i in range(atom_ct):
                pts = f.readline().strip()
                pts = pts.split()
                atoms.append((pts[0], i))
                tmp[:3] = np.asarray(pts[1:], dtype=np.float64)
                nbrs[i] = tmp.copy()
            atom_pts[n] = nbrs
            a = f.readline().split()
            atom_ct = int(a[0])
            nbrs = np.resize(nbrs, [atom_ct, 3])
            f.readline()

        return atom_pts

    def single_cell(self, f):
        """Read xyz file for point set registration.

        f - open file for reading in xyz format
        returns:
        atom_pts - n x 4 np.array of atoms and placement
        of the form: atom (as int! placement in .xyz file) x y z
        atoms - dictionary with elements as keys
        and each value a list with the elements position in
        the .xyz file.
        """

        a = f.readline().split()  # number of atoms in configuration
        atom_ct = int(a[0])
        atom_pts = np.empty([atom_ct, 3])
        tmp = np.empty(3)
        f.readline()

        for i in range(atom_ct):
            pts = f.readline().strip()
            pts = pts.split()
            tmp = np.asarray(pts[1:], dtype=np.float64)
            atom_pts[i] = tmp

        return atom_pts


class makeDiagrams():
    """Create persistence diagrams for materials classification."""
    def __init__(self, bcc_len=0, fcc_len=0, test_len=0):
        if test_len == 0:
            self.full_len = bcc_len + fcc_len
        else:
            self.full_len = test_len
        self.bcc_len = bcc_len
        self.fcc_len = fcc_len
        self.bcc_cells = []
        self.fcc_cells = []
        self.fcc_dgms = []
        self.bcc_dgms = []
        self.cells = []
        self.dgms = [None] * self.full_len

    def find_inf(self, dgm):
        """Remove 0-dim feature with infty as death time."""
        idx = np.where(dgm[0] == np.inf)[0]
        dgm[0] = np.delete(dgm[0], obj=idx, axis=0)
        return dgm

    def make_bcc_dgms(self):
        tmp = [ripser(cell, maxdim=2, thresh=5)['dgms']
               for cell in self.bcc_cells]
        self.bcc_dgms = [self.find_inf(dgm) for dgm in tmp]

    def make_fcc_dgms(self):
        tmp = [ripser(cell, maxdim=2, thresh=5)['dgms']
               for cell in self.fcc_cells]
        self.fcc_dgms = [self.find_inf(dgm) for dgm in tmp]

    def make_dgms(self):
        tmp = [ripser(cell, maxdim=2, thresh=5)['dgms']
               for cell in self.cells]
        self.dgms = [self.find_inf(dgm) for dgm in tmp]

    def dgm_lists(self):
        self.dgms = [dgm for dgm in self.bcc_dgms]
        self.dgms.extend(self.fcc_dgms)
        return None

    def clear(self):
        self.bcc_cells[:] = []
        self.fcc_cells[:] = []
        self.fcc_dgms[:] = []
        self.bcc_dgms[:] = []
        self.dgms[:] = []
