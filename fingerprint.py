#!/usr/bin/env python3

import numpy as np
import utils
import dist


def main():
    """Create feature matrix for materials fingerprinting method."""
    print("\n Process Started")
    utils.timestamp()

    num_b = 10
    num_f = 10
    # total = num_b + num_f
    data = utils.makeDiagrams(num_b, num_f)
    data_pts = utils.getData()
    datafile = 'cells.xyz'

    print(' Total PDs = {:4d}'.format(num_b + num_f))

    print("\n Reading Training Data")
    data_pts.read_data(datafile, data)
    data.make_bcc_dgms()
    data.make_fcc_dgms()
    data.dgm_lists()
    distance = dist.distances(data, metric='dpc')

    print("\n Multiprocessing Distances\n\n Training Set")
    for k in range(3):
        distance.compute_bcc(data, k)
        distance.compute_fcc(data, k)
    # distance.dists_mp(data)
    distance.feature_matrix()
    np.save('repo_test.out', distance.X)

    utils.timestamp()
    return distance


if __name__ == "__main__":
    main()
