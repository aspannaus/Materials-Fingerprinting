#!/usr/bin/env python3

import numpy as np
from ripser import ripser
from os.path import exists
import time


def timestamp(f=None):
    t = time.time()
    if f is not None:
        f.write(time.ctime(t))
    else:
        str = ' ' + time.ctime()
        print(str)
    return None


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

    def read_data(self, in_file, dgms):
        str = '/Users/spannaus/Documents/GitHub/APT_analysis/real_data/bcc_'
        self.bcc_file = str + in_file
        self.get_file(self.bcc_file)
        str = '/Users/spannaus/Documents/GitHub/APT_analysis/real_data/fcc_'
        self.fcc_file = str + in_file
        self.get_file(self.fcc_file)

        with open(self.bcc_file) as f_in:
            print(" Reading data from {:s}".format(self.bcc_file))
            dgms.bcc_cells = self.read_file(f_in, dgms.bcc_len)
        with open(self.fcc_file) as f_in:
            print(" Reading data from {:s}".format(self.fcc_file))
            dgms.fcc_cells = self.read_file(f_in, dgms.fcc_len)
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


class makeDiagrams():
    """Create persistence diagrams for materials classification."""

    def __init__(self, bcc_len, fcc_len):
        self.full_len = bcc_len + fcc_len
        self.bcc_len = bcc_len
        self.fcc_len = fcc_len
        self.bcc_cells = []
        self.fcc_cells = []
        self.fcc_dgms = []
        self.bcc_dgms = []
        self.dgms = []

    def find_inf(self, dgm):
        """Remove 0-dim feature with infinity as death time."""
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
        return None
