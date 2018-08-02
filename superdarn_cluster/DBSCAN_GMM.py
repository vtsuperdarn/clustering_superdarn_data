import numpy as np
from sklearn.cluster import DBSCAN
import copy
from superdarn_cluster.time_utils import time_days_to_index
from superdarn_cluster.utilities import blanchard_gs_flg
from sklearn.mixture import BayesianGaussianMixture
from superdarn_cluster.utilities import plot_clusters
from scipy.stats import boxcox

# TODO this one and dbscan.ipynb are very similar. Which one is up to date? Should I keep both? Does dbscan.ipynb still work?
# TODO maybe turn this script into a module in superdarn_cluster, like STDBSCAN etc. Keep the plotting code for param tuning.
# This script is really useful because it has some extra features for visualization that plot_fanplots_rtiplots doesn't do -
# specifically the cluster plots, for DBSCAN and then for DBSCAN + GMM. You can't otherwise tell which clusters are from GMM / DBSCAN,
# unless you use some special code (like negative cluster #s).
# Either update this script to match newer scripts, or update fanplots_rtiplots to also plot individual clusters (on RTI).

class DBSCAN_GMM:

    def __init__(self, params, alg='GMM', cluster_factor=10):
        self.time_eps = params['time_eps']
        self.beam_eps = params['beam_eps']
        self.gate_eps = params['gate_eps']
        self.eps = params['eps']
        self.min_pts = params['min_pts']
        self.n_clusters = params['n_clusters']
        self.alg = alg
        assert cluster_factor > self.n_clusters      # This is used to label GMM clusters created from DBSCAN clusters
        self.cluster_factor = cluster_factor

    def fit(self, beam, gate, time, vel, wid):
        """ Do DBSCAN """
        time_integ = time_days_to_index(time)  # integration time, float
        X = np.column_stack((beam / self.beam_eps, gate / self.gate_eps, time_integ / self.time_eps))
        db = DBSCAN(eps=self.eps, min_samples=self.min_pts).fit(X)
        labels = db.labels_
        db_labels_unique = np.unique(labels)
        print('DBSCAN clusters: '+str(np.max(db_labels_unique)))

        """ Do GMM """
        # ~~ BoxCox on velocity and width
        bx_vel, h_vel = boxcox(np.abs(vel))
        bx_wid, h_wid = boxcox(np.abs(wid))
        gmm_data = np.column_stack((bx_vel, bx_wid, time_integ, gate, beam))

        for dbc in db_labels_unique:
            db_cluster_mask = (labels == dbc)
            if dbc == -1:
                continue
            num_pts = np.sum(db_cluster_mask)
            # Sometimes DBSCAN will find tiny clusters due to this:
            # https://stackoverflow.com/questions/21994584/can-the-dbscan-algorithm-create-a-cluster-with-less-than-minpts
            # I don't want to keep these clusters, so label them as noise
            if num_pts < self.min_pts:
                labels[db_cluster_mask] = -1
            # TODO base this on variance(abs(whatev, maybe vel, wid, range gate, power)))
            if num_pts < 500:
                continue
            if self.alg == 'GMM':
                data = gmm_data[db_cluster_mask]
                # Using 3 components will hopefully separate it into 3 groups: IS, GS, and Noise (high variance)
                # Or maybe it will create 2 IS clusters and 1 GS
                # It's not perfect, but BayesGMM doesn't behave well at higher number of clusters.
                # Perhaps the number of clusters used should actually depend on the size of the dataset
                estimator = BayesianGaussianMixture(n_components=self.n_clusters,
                                                    covariance_type='full', max_iter=500,
                                                    random_state=0, n_init=5, init_params='kmeans',
                                                    weight_concentration_prior=1,
                                                    weight_concentration_prior_type='dirichlet_process')
                estimator.fit(data)
                gmm_labels = estimator.predict(data)
                #gmm_labels = - gmm_labels - dbc * self.cluster_factor  # e.g., DBSCAN cluster 3 will be broken up into GMM clusters -31, -32, -33
                gmm_labels += np.max(labels) + 1
                labels[db_cluster_mask] = gmm_labels    # TODO make sure this works okay
            elif self.alg == 'k-means':
                raise('k-means not yet implememnted')
            else:
                raise('bad alg '+str(self.alg))

        return labels


if __name__ == '__main__':
    # ~~ Get data ~~
    from superdarn_cluster.dbtools import flatten_data_11_features, read_db
    import datetime as dt

    start_time = dt.datetime(2018, 2, 7)
    end_time = dt.datetime(2018, 2, 8)
    rad = 'sas'
    db_path = "./Data/sas_GSoC_2018-02-07.db"
    num_beams = 16
    b = 0
    data_dict = read_db(db_path, rad, start_time, end_time)
    data_flat_unscaled = flatten_data_11_features(data_dict, remove_close_range=True)

    import matplotlib.pyplot as plt
    from sklearn.preprocessing import scale

    feature_names = ['beam', 'gate', 'vel', 'wid', 'power', 'freq', 'time', 'phi0', 'elev', 'nsky', 'nsch']

    gate = data_flat_unscaled[:, 1]
    power = data_flat_unscaled[:, 4]
    beam = data_flat_unscaled[:, 0]
    vel = data_flat_unscaled[:, 2]
    wid = data_flat_unscaled[:, 3]
    time_num_days = data_flat_unscaled[:, 6]

    from matplotlib.dates import date2num

    # What matters for scaling this is the size of each step between these (discrete) measurements.
    # If you want to connect things within 1 range gate and 1 beam, do no scaling and set eps ~= 1.1
    # If you want to connect things within 6 time measurements, scale it so that 6 * dt = 1 and eps ~= 1.1
    # Time has some gaps in between each scan of 16 beams, so epsilon should be large enough
    # TODO this with the time_utils functions
    scaled_time = (time_num_days - time_num_days[0])  # (time - np.floor(time)) * 24 * 60 * 60
    uniq_time = np.sort(np.unique(scaled_time))
    shifted_time = np.roll(uniq_time, -1)
    dt = np.min((shifted_time - uniq_time)[:-1])
    print(dt)
    integer_time = scaled_time / dt
    scaled_time = scale(scaled_time / (dt))
    # Divide by variance and center mean at 0
    scaled_gate = gate
    scaled_beam = beam

    # ~~ DBSCAN ~~
    # ~~ Important note: On certain systems (HDD + 8GB RAM + Ubuntu16 desktop) this won't run due to memory consumption
    # ~~ Works fine on my laptop (SSD + 8GB RAM + Ubuntu18 + decent i5)
    time_integ = time_days_to_index(time_num_days)
    time_eps = 20.0           # 1 scan ish
    beam_eps = 3.0
    gate_eps = 1.0

    X = np.column_stack((beam / beam_eps, gate / gate_eps, time_integ / time_eps))
    print(X.shape)

    eps, minPts = 1, 5      #TODO think abt raising this to 10 or so
    db = DBSCAN(eps=eps, min_samples=minPts).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    #db = DBSCAN_GMM(time_eps, beam_eps, gate_eps, eps, minPts, n_clusters=3)
    #labels = db.fit({'time': time_num_days, 'beam': beam, 'gate': gate, 'vel': vel, 'wid': wid})


    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('DBSCAN clusters: %d' % n_clusters_)

    unique_labels = list(np.unique(labels))
    unique_labels.remove(-1)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, np.max(unique_labels)+1)]
    np.random.shuffle(colors)
    range_max = data_dict['nrang'][0]

    from matplotlib.dates import date2num

    # For each unique time unit
    times_unique_dt = data_dict['datetime']
    times_unique_index = np.array([date2num(d) for d in data_dict['datetime']])
    from superdarn_cluster.time_utils import *

    times_unique_index = time_days_to_index(times_unique_index)
    index_time = time_days_to_index(time_num_days)

    # ~~ Fanplots ~~
    # Note this will create >1000 plots for 1 day of data, so it takes a while (10 minutes maybe).
    """ """

    from superdarn_cluster.FanPlot import FanPlot
    fan_colors = list(colors)
    fan_colors.append((0, 0, 0, 1))
    fanplot = FanPlot()
    fanplot.plot_all(times_unique_dt, times_unique_index, index_time, beam, gate, labels, fan_colors)

    # ~~ Plotting all DBSCAN clusters on RTI plot (scatterplot) ~~
    for b in range(num_beams):
        fig = plt.figure(figsize=(16, 8))
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)
            beam_mask = (beam == b)

            xy = X[class_member_mask & beam_mask]
            plt.plot(xy[:, 2], xy[:, 1], '.', color=tuple(col), markersize=10)


        plt.xlim((np.min(integer_time / time_eps), np.max(integer_time / time_eps)))
        plt.ylim((np.min(gate / gate_eps), np.max(gate / gate_eps)))
        plt.title('Beam %d \n Clusters: %d   Eps: %.2f   MinPts: %d ' % (b, n_clusters_, eps, minPts))
        # plt.show()
        plt.savefig('dbscan beam ' + str(b))
        plt.close()

    from superdarn_cluster.utilities import plot_clusters

    stats_i = [0, 1, 2, 3, 4, 7, 8]
    data_flat_unscaled[:, 2] = np.abs(data_flat_unscaled[:, 2])
    data_flat_unscaled[:, 3] = np.abs(data_flat_unscaled[:, 3])
    plot_clusters(labels, data_flat_unscaled[:, stats_i], data_flat_unscaled[:, 6],
                  gate, vel, np.array(feature_names)[stats_i], range_max, start_time, end_time, save=True,
                  base_path='dbscan ')

    # ~~ GMM ~~
    from sklearn.mixture import BayesianGaussianMixture
    from superdarn_cluster.utilities import plot_clusters
    from scipy.stats import boxcox

    # ~~ BoxCox on velocity and width
    bx_vel, h_vel = boxcox(np.abs(vel))
    bx_wid, h_wid = boxcox(np.abs(wid))
    gmm_data = np.column_stack((bx_vel, bx_wid, time_num_days, gate, beam))

    def get_gs_flg(v, w):
        med_vel = np.median(np.abs(v))
        med_wid = np.median(np.abs(w))
        # return med_vel < 33.1 + 0.139 * med_wid - 0.00133 * (med_wid ** 2)
        # return med_vel < 30 - med_wid * 1.0 / 3.0
        return med_vel < 15

    stats_i = [0, 1, 2, 3, 4, 7, 8]
    gs_flg = np.zeros(len(time_num_days))
    for k in unique_labels:
        class_member_mask = (labels == k)
        if k == -1:
            gs_flg[class_member_mask] = -1
            continue
        k_vel = vel[class_member_mask]
        k_wid = wid[class_member_mask]
        gs_flg[class_member_mask] = get_gs_flg(k_vel, k_wid)


    # ~~ IS/GS Colormesh RTI ~~
    from superdarn_cluster.utilities import plot_is_gs_colormesh

    # fig = plt.figure(figsize=(16, 4))
    # ax = plt.subplot(111)
    unique_time = [date2num(d) for d in data_dict['datetime']]

    for b in range(num_beams):
        fig = plt.figure(figsize=(16, 4))
        ax = plt.subplot(111)
        beam_filter = b == beam
        plot_is_gs_colormesh(ax, unique_time, time_num_days[beam_filter], gate[beam_filter], gs_flg[beam_filter], range_max,
                             plot_indeterminate=True)
        plt.title('gs is colormesh code threshold BoxCox beam ' + str(b))
        plt.savefig('gs is colormesh code threshold BoxCox beam ' + str(b))

