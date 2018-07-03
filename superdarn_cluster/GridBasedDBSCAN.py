# -*- coding: utf-8 -*-

# A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise
# Martin Ester, Hans-Peter Kriegel, Jörg Sander, Xiaowei Xu
# dbscan: density based spatial clustering of applications with noise
# based on:

import numpy as np
import math
from matplotlib.dates import date2num
from superdarn_cluster.time_utils import *
import matplotlib.pyplot as plt
from superdarn_cluster.FanPlot import FanPlot



UNCLASSIFIED = False
NOISE = -1


class GridBasedDBSCAN():

    def __init__(self, gate_eps, beam_eps, time_eps, min_pts, nrang, nbeam, dr, dtheta, r_init=0):
        dtheta = dtheta * np.pi / 180.0
        self.C = np.zeros((nrang, nbeam))
        for i in range(nrang):
            for j in range(nbeam):
                # This is the ratio between radial and angular distance for each point. Across a row it's all the same, consider removing j.
                self.C[i,j] = self._calculate_ratio(dr, dtheta, i, j, r_init=r_init)
        print(self.C)
        print()
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

        h = space_eps[0]
        w = space_eps[1]

        # Search in an ellipsoid with the 3 epsilon values (slower, performs worse than filter so far)
        # t = self.time_eps
        # in_ellipse = ((q[0] - p[0])**2 / w**2 + (q[1] - p[1])**2 / h**2 + (q[2] - p[2])**2 / t**2) <= 1

        # Search in an ellipse with widths defined by the 2 epsilon values
        in_ellipse = ((q[0] - p[0])**2 / h**2 + (q[1] - p[1])**2 / w**2) <= 1

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
    # TODO why does this make a matrix if it doesn't vary from beam to beam
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


        # TODO adaptive minPts??
        g, f = self.beam_eps, 1

        #w0 = 1
        cluster_id = 1
        n_points = m.shape[1]
        classifications = [UNCLASSIFIED] * n_points
        for point_id in range(0, n_points):
            point = m[:, point_id]
            i, j = int(point[0]), int(point[1]) # range gate, beam
            wid = g / (f * self.C[i, j])
            eps = (self.gate_eps, wid)
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
    end_time = dt.datetime(2018, 2, 7, 14, 15)
    rad = 'sas'
    db_path = "../Data/sas_GSoC_2018-02-07.db"
    num_beams = 16
    num_gates = 75
    b = 0
    data_dict = read_db(db_path, rad, start_time, end_time)
    data_flat_unscaled = flatten_data_11_features(data_dict, remove_close_range=False)

    gate = data_flat_unscaled[:, 1]
    beam = data_flat_unscaled[:, 0]
    time = data_flat_unscaled[:, 6]
    time_sec = time_days_to_sec(time)
    time_index = time_sec_to_index(time_sec)

    data = np.column_stack((gate, beam, time_index)).T

    #NOTE - these params need to change if you set remove_close_range=False. Params determined on 15min often work for longer time periods.
    gdb = GridBasedDBSCAN(gate_eps=3.0, beam_eps=4.0, time_eps=30, min_pts=5, nrang=75, nbeam=16, dr=45, dtheta=3.3, r_init=180)
    labels = gdb.fit(data)
    labels = np.array(labels)
    clusters = np.unique(labels)

    colors = plt.cm.plasma(np.linspace(0, 1, len(clusters)))
    colors[0] = [0, 0, 0, 1] #plot noise black
    plt.figure(figsize=(16, 8))
    print('clusters: ', clusters)

    for b in range(num_beams):
        beam_mask = beam == b
        data_b = data[:, beam_mask]
        labels_b = labels[beam_mask]
        if not data_b.any(): continue
        for i, label in enumerate(clusters):
            plt.scatter(data_b[2, labels_b == label], data_b[0, labels_b == label], color=colors[i])
        plt.savefig("../grid-based DBSCAN RTI beam " + str(b))
        plt.close()

    # For each unique time unit
    times_unique_dt = data_dict['datetime']
    times_unique_num = [date2num(t) for t in data_dict['datetime']]
    labels = np.array(labels)
    print(len(times_unique_dt))

    fan_colors = list(colors)
    fan_colors.append((0, 0, 0, 1))
    fanplot = FanPlot()
    fanplot.plot_all(data_dict['datetime'], np.unique(time_index), time_index, beam, gate, labels, fan_colors,
                     base_path="../grid-based dbscan ")


    # TEST : epsilon division trick
    from superdarn_cluster.DBSCAN import dbscan

    def _calculate_ratio(dr, dt, i, j, r_init=0):
        r_init, dr, dt, i, j = float(r_init), float(dr), float(dt), float(i), float(j)
        cij = (r_init + dr * i) / (2.0 * dr) * (np.sin(dt * (j + 1.0) - dt * j) + np.sin(dt * j - dt * (j - 1.0)))
        return cij


    # TODO beam_eps needs to be high enough s.t. at gate 75 its >= 1
    beam_eps = 4.0 / np.array([_calculate_ratio(45, 3.3 * np.pi / 180, g, 0, r_init=180) for g in range(num_gates)])
    print( np.array([_calculate_ratio(45, 3.3 * np.pi / 180, g, 0, r_init=180) for g in range(num_gates)]))
    gate_eps = 3.0
    time_eps = 30

    beam_x = [beam[i] / beam_eps[int(gate[i])] for i in range(len(beam))]

    X = np.column_stack((beam_x, gate / gate_eps, time_index / time_eps)).T

    eps, min_pts = 1, 5
    labels = dbscan(X, eps=eps, min_points=min_pts)
    labels = np.array(labels)

    fan_colors = list(colors)
    fan_colors.append((0, 0, 0, 1))
    fanplot = FanPlot()
    fanplot.plot_all(data_dict['datetime'], np.unique(time_index), time_index, beam, gate, labels, fan_colors,
                     base_path="../regular dbscan ")


    #core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    #core_samples_mask[db.core_sample_indices_] = True
    #labels = db.labels_

    """
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('DBSCAN clusters: %d' % n_clusters_)

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    range_max = data_dict['nrang'][0]

    # ~~ Plotting all DBSCAN clusters on RTI plot (scatterplot) ~~
    for b in range(num_beams):
        fig = plt.figure(figsize=(16, 8))
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)
            beam_mask = (beam == b)

            xy = X[class_member_mask & core_samples_mask & beam_mask]
            plt.plot(xy[:, 2], xy[:, 1], '.', color=tuple(col), markersize=10)

            xy = X[class_member_mask & ~core_samples_mask & beam_mask]
            plt.plot(xy[:, 2], xy[:, 1], '.', color=tuple(col), markersize=10)

        plt.xlim((np.min(time_index / time_eps), np.max(time_index / time_eps)))
        plt.ylim((np.min(gate / gate_eps), np.max(gate / gate_eps)))
        plt.title('Beam %d \n Clusters: %d   Eps: %.2f   MinPts: %d ' % (b, n_clusters_, eps, min_pts))
        # plt.show()
        plt.savefig('../regular dbscan beam ' + str(b))
        plt.close()


    fan_colors = list(colors)
    fan_colors.append((0, 0, 0, 1))
    fanplot = FanPlot()
    fanplot.plot_all(data_dict['datetime'], np.unique(time_index), time_index, beam, gate, labels, fan_colors, base_path="../regular dbscan ")
    """




