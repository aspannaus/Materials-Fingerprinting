#!/usr/bin/env python3
# file: classify_utils.py
# author: Adam Spannaus
# date: 12/08/2020
# utility functions and classes for materials fingerprinting code

import numpy as np
from ripser import ripser
from os.path import exists, expanduser
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


class getData():
    """Class to fetch and read in data.

        Methods
        ___________
        get_file(in_file)
            checks that data file exists and prompts user if not

        read_data(f_name, dgms)
            opens input file and saves data

        read_file
            reads input files
    """
    def __init__(self):
        pass

    def get_file(self, in_file):
        """Checks that input file exists and prompts user for vaild filename if not."""
        while not(exists(in_file)):
            print(" File {:s} does not exist. Enter a valid filename.\n".
                  format(in_file))
            in_file = input("--> ")
        return in_file

    def read_data(self, f_name, dgms):
        """Reads data and saves it for further processing.

        params
        ___________
        f_name: base datafile name, ie. BCC, FCC, or HCP

        dgms: diagrams object that stores neighborhods for persistence diagrams

        """
        path = expanduser('~/Desktop/Materials-Fingerprinting/data/')
        # path = '/Users/Shared/ornldev/projects/GitHub/Materials-Fingerprinting/data/'

        if f_name[0] == 'B':
            self.bcc_file = path + f_name
            if not(exists(self.bcc_file)):
                self.bcc_file = self.get_file(self.bcc_file)
            with open(self.bcc_file) as f_in:
                print(" Reading data from {:s}".format(self.bcc_file))
                dgms.bcc_cells = self.read_file(f_in, dgms.bcc_len)
        elif f_name[0] == 'F':
            self.fcc_file = path + f_name
            if not(exists(self.fcc_file)):
                self.fcc_file = self.get_file(self.fcc_file)
            with open(self.fcc_file) as f_in:
                print(" Reading data from {:s}".format(self.fcc_file))
                dgms.fcc_cells = self.read_file(f_in, dgms.fcc_len)
        elif f_name[0] == 'H':
            self.hcp_file = path + f_name
            if not(exists(self.hcp_file)):
                self.hcp_file = self.get_file(self.hcp_file)
            with open(self.hcp_file) as f_in:
                print(" Reading data from {:s}".format(self.hcp_file))
                dgms.hcp_cells = self.read_file(f_in, dgms.hcp_len)

        return None

    def read_file(self, f, n_config):
        """Read xyz file for point set registration.

        f - open file for reading in xyz format
        returns:
        atom_pts - n x 4 list of atoms and placement
        of the form: atom (as int! placement in .xyz file) x y z
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
    """Create persistence diagrams for materials classification.

    Methods
    ______________

    find_inf:
        finds inf value in persistence diagram and removes it

    make_bcc_dgms:
        makes diagrams from bcc data

    make_fcc_dgms:
        makes diagrams from bcc data

    make_hcp_dgms:
        makes diagrams from bcc data

    dgm_lists:
        concatenates diagrams into one list

    clear:
        deletes contents of dgm and cell lists

    """
    def __init__(self, bcc_len, fcc_len, hcp_len=0):
        self.full_len = bcc_len + fcc_len + hcp_len
        self.bcc_len = bcc_len
        self.fcc_len = fcc_len
        self.hcp_len = hcp_len
        if self.bcc_len > 0:
            self.bcc_dgms = []
            self.bcc_cells = []
        if self.fcc_len > 0:
            self.fcc_dgms = []
            self.fcc_cells = []
        self.dgms = []
        if hcp_len > 0:
            self.hcp_dgms = []
            self.hcp_cells = []

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

    def make_hcp_dgms(self):
        tmp = [ripser(cell, maxdim=2, thresh=5)['dgms']
               for cell in self.hcp_cells]
        self.hcp_dgms = [self.find_inf(dgm) for dgm in tmp]

    def dgm_lists(self):
        if self.bcc_len > 0:
            self.dgms.extend(self.bcc_dgms)
        if self.fcc_len > 0:
            self.dgms.extend(self.fcc_dgms)
        if self.hcp_len > 0:
            self.dgms.extend(self.hcp_dgms)
        return None

    def clear(self):
        if self.bcc_len > 0:
            self.bcc_cells[:] = []
            self.bcc_dgms[:] = []
        if self.fcc_len > 0:
            self.fcc_cells[:] = []
            self.fcc_dgms[:] = []
        if self.hcp_len > 0:
            self.hcp_dgms[:] = []
            self.hcp_cells[:] = []
        self.dgms[:] = []
