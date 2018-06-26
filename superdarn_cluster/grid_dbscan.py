# -*- coding: utf-8 -*-

# A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise
# Martin Ester, Hans-Peter Kriegel, Jörg Sander, Xiaowei Xu
# dbscan: density based spatial clustering of applications with noise

import numpy as np
import math

UNCLASSIFIED = False
NOISE = -1


class GridBasedDBSCAN():

    def __init__(self, gate_eps, beam_eps, time_eps, min_pts, nrang, nbeam, dr, dtheta, r_init=0):
        dtheta = dtheta * 3.14159 / 360.0
        self.C = np.zeros((nrang, nbeam))
        for i in range(nrang):
            for j in range(nbeam):
                # This is the ratio between radial and angular distance for each point. Across a row it's all the same, consider removing j.
                self.C[i,j] = self._calculate_ratio(dr, dtheta, i, j, r_init=r_init)
        self.gate_eps = gate_eps
        self.beam_eps = beam_eps
        self.time_eps = time_eps
        self.min_pts = min_pts


    def _eps_neighborhood(self, p, q, space_eps):
        # filter by time neighbors

        min_time = p[2] - self.time_eps
        max_time = p[2] + self.time_eps
        time_neighbor = q[2] >= min_time and q[2] <= max_time
        if not time_neighbor:
            return False


        # TODO change this to use class variables
        w = space_eps[0]
        h = space_eps[1]
        t = self.time_eps
        # Search in an ellipsoid with the 3 epsilon values (slower, performs similar to the filter so far)
        # in_ellipse = ((q[0] - p[0])**2 / w**2 + (q[1] - p[1])**2 / h**2 + (q[2] - p[2])**2 / t**2) <= 1
        # Search in an ellipse with widths defined by the 2 epsilon values
        in_ellipse = ((q[0] - p[0])**2 / w**2 + (q[1] - p[1])**2 / h**2) <= 1
        return in_ellipse


    def _region_query(self, m, point_id, eps):
        n_points = m.shape[1]
        seeds = []
        for i in range(0, n_points):
            if self._eps_neighborhood(m[:, point_id], m[:, i], eps):
                seeds.append(i)
        return seeds


    def _expand_cluster(self, m, classifications, point_id, cluster_id, eps, min_points):
        seeds = self._region_query(m, point_id, eps)
        if len(seeds) < min_points:
            classifications[point_id] = NOISE
            return False
        else:
            classifications[point_id] = cluster_id
            for seed_id in seeds:
                classifications[seed_id] = cluster_id

            while len(seeds) > 0:
                current_point = seeds[0]
                results = self._region_query(m, current_point, eps)
                if len(results) >= min_points:
                    for i in range(0, len(results)):
                        result_point = results[i]
                        if classifications[result_point] == UNCLASSIFIED or \
                                classifications[result_point] == NOISE:
                            if classifications[result_point] == UNCLASSIFIED:
                                seeds.append(result_point)
                            classifications[result_point] = cluster_id
                seeds = seeds[1:]
            return True


    # Input for grid-based DBSCAN:
    # C matrix calculated based on sensor data.
    # Based on Birant et al
    def _calculate_ratio(self, dr, dt, i, j, r_init=0):
        r_init, dr, dt, i, j = float(r_init), float(dr), float(dt), float(i), float(j)
        cij = (r_init + dr * i) / (2.0 * dr) * (np.sin(dt * (j + 1.0) - dt * j) + np.sin(dt * j - dt * (j - 1.0)))
        return cij


    def fit(self, m):
        """Implementation of Density Based Spatial Clustering of Applications with Noise
        See https://en.wikipedia.org/wiki/DBSCAN

        scikit-learn probably has a better implementation

        Uses Euclidean Distance as the measure

        Inputs:
        m - A matrix whose columns are feature vectors
        eps - Maximum distance two points can be to be regionally related
        min_points - The minimum number of points to make a cluster

        Outputs:
        An array with either a cluster id number or dbscan.NOISE (-1) for each
        column vector in m.
        """
        # These should be fractional #'s i think? Not 100% sure...
        # I think this shouldn't be in units of km.
        # At further ranges, the ellipse should become less wide. That's what this should be doing. Adaptive elliptical search area.


        # TODO adaptive minPts
        g, f = self.beam_eps, 1

        #w0 = 1
        cluster_id = 1
        n_points = m.shape[1]
        classifications = [UNCLASSIFIED] * n_points
        for point_id in range(0, n_points):
            point = m[:, point_id]
            i, j = int(point[0]), int(point[1]) # range gate, beam
            wid = g / (f * self.C[i, j])
            eps = (wid, self.gate_eps)
            # Adaptively change one of the epsilon values and the min_points parameter using the C matrix
            if classifications[point_id] == UNCLASSIFIED:
                if self._expand_cluster(m, classifications, point_id, cluster_id, eps, self.min_pts):
                    cluster_id = cluster_id + 1
        return classifications


if __name__ == "__main__":
    import numpy as np
    from superdarn_cluster.dbtools import flatten_data_11_features, read_db
    import datetime as dt

    start_time = dt.datetime(2018, 2, 7, 14)
    end_time = dt.datetime(2018, 2, 7, 16)
    rad = 'sas'
    db_path = "../Data/sas_GSoC_2018-02-07.db"
    b = 0
    data_dict = read_db(db_path, rad, start_time, end_time)
    data_flat_unscaled = flatten_data_11_features(data_dict, remove_close_range=False)

    gate = data_flat_unscaled[:, 1]
    beam = data_flat_unscaled[:, 0]

    time = data_flat_unscaled[:, 6]
    scaled_time = (time - time[0]) #(time - np.floor(time)) * 24 * 60 * 60
    uniq_time = np.sort(np.unique(scaled_time))
    shifted_time = np.roll(uniq_time, -1)
    dt = np.min((shifted_time - uniq_time)[:-1])
    integer_time = scaled_time / (dt)


    data = np.column_stack((gate, beam, integer_time)).T
    #print(data.shape)
    #data = np.unique(data, axis=1)
    #print(data.shape)
    print(data[:, :20])

    gdb = GridBasedDBSCAN(gate_eps=2.0, beam_eps=3.0, time_eps=30, min_pts=20, nrang=75, nbeam=16, dr=45, dtheta=3.3, r_init=180)
    labels = gdb.fit(data)
    clusters = np.unique(labels)

    import matplotlib.pyplot as plt
    colors = plt.cm.plasma(np.linspace(0, 1, len(clusters)))
    print('clusters: ', clusters)
    for i, label in enumerate(clusters):
        if label == -1:
            color = [0, 0, 0, 1]        # black for noise
        else:
            color = colors[i]
        plt.scatter(data[2, labels == label], data[0, labels == label], color=color)
    plt.show()



