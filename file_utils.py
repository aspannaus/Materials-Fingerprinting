import numpy as np


class readData():
    def __init__(self, f_in):
        self.f = f_in

    def single_cell(self):
        """Read xyz file for point set registration.

        f - open file for reading in xyz format
        returns:
        atom_pts - n x 4 np.array of atoms and placement
        of the form: atom (as int! placement in .xyz file) x y z
        atoms - dictionary with elements as keys
        and each value a list with the elements position in
        the .xyz file.
        """

        a = self.f.readline().split()  # number of atoms in configuration
        atom_ct = int(a[0])
        atom_pts = np.empty([atom_ct, 3])
        tmp = np.empty(3)
        self.f.readline()

        for i in range(atom_ct):
            pts = self.f.readline().strip()
            pts = pts.split()
            tmp = np.asarray(pts[1:], dtype=np.float64)
            atom_pts[i] = tmp
        return atom_pts

    def read_file(self, n_config):
        """Read xyz file for point set registration.

        f - open file for reading in xyz format
        returns:
        atom_pts - n x 4 np.array of atoms and placement
        of the form: atom (as int! placement in .xyz file) x y z
        atoms - dictionary with elements as keys
        and each value a list with the elements position in
        the .xyz file.
        """
        atoms = []

        a = self.f.readline().split()  # number of atoms in configuration
        atom_ct = int(a[0])
        nbrs = np.empty([atom_ct, 3])
        atom_pts = [None] * n_config
        tmp = np.empty(3)
        self.f.readline()

        for n in range(n_config):
            for i in range(atom_ct):
                pts = self.f.readline().strip()
                pts = pts.split()
                atoms.append((pts[0], i))
                tmp[:3] = np.asarray(pts[1:], dtype=np.float64)
                nbrs[i] = tmp.copy()
            atom_pts[n] = nbrs
            a = self.f.readline().split()
            atom_ct = int(a[0])
            nbrs = np.resize(nbrs, [atom_ct, 3])
            self.f.readline()
        return atom_pts


def read_file(f_in, n_config):
    """Read xyz file for point set registration.

    f - open file for reading in xyz format
    returns:
    atom_pts - n x 4 np.array of atoms and placement
    of the form: atom (as int! placement in .xyz file) x y z
    atoms - dictionary with elements as keys
    and each value a list with the elements position in
    the .xyz file.
    """
    atoms = []

    a = f_in.readline().split()  # number of atoms in configuration
    atom_ct = int(a[0])
    nbrs = np.empty([atom_ct, 3])
    atom_pts = [None] * n_config
    tmp = np.empty(3)
    f_in.readline()

    for n in range(n_config):
        for i in range(atom_ct):
            pts = f_in.readline().strip()
            pts = pts.split()
            atoms.append((pts[0], i))
            tmp[:3] = np.asarray(pts[1:], dtype=np.float64)
            nbrs[i] = tmp.copy()
        atom_pts[n] = nbrs
        a = f_in.readline().split()
        atom_ct = int(a[0])
        nbrs = np.resize(nbrs, [atom_ct, 3])
        f_in.readline()
    return atom_pts
