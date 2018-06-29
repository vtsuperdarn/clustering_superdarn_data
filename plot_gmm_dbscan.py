
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.cluster import DBSCAN

# ~~ Get data ~~
from superdarn_cluster.dbtools import flatten_data_11_features, read_db
import datetime as dt

start_time = dt.datetime(2018, 2, 7)
end_time = dt.datetime(2018, 2, 8)
rad = 'sas'
db_path = "./Data/sas_GSoC_2018-02-07.db"
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

# What matters for scaling this is the size of each step between these (discrete) measurements.
# If you want to connect things within 1 range gate and 1 beam, do no scaling and set eps ~= 1.1
# If you want to connect things within 6 time measurements, scale it so that 6 * dt = 1 and eps ~= 1.1
# Time has some gaps in between each scan of 16 beams, so epsilon should be large enough
# TODO this with the time_utils functions
scaled_time = (time_num_days - time_num_days[0]) #(time - np.floor(time)) * 24 * 60 * 60
uniq_time = np.sort(np.unique(scaled_time))
shifted_time = np.roll(uniq_time, -1)
dt = np.min((shifted_time - uniq_time)[:-1])
integer_time = scaled_time / dt
scaled_time = scale(scaled_time / (dt))
# Divide by variance and center mean at 0
scaled_gate = gate
scaled_beam = beam

# ~~ DBSCAN ~~
# ~~ Important note: On certain systems (HDD + 8GB RAM + Ubuntu16 desktop) this won't run due to memory consumption
# ~~ Works fine on my laptop (SSD + 8GB RAM + Ubuntu18, decent i5)
time_eps = 50.0
beam_eps = 2.0
gate_eps = 3.0

X = np.column_stack((beam / beam_eps, gate / gate_eps, integer_time / time_eps))
print(X.shape)

eps, minPts = 1, 15
db = DBSCAN(eps=eps, min_samples=minPts).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('DBSCAN clusters: %d' % n_clusters_)

unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
range_max = data_dict['nrang'][0]

from matplotlib.dates import date2num
# For each unique time unit
times_unique_dt = data_dict['datetime']
times_unique_index = np.array([date2num(d) for d in data_dict['datetime']])
from superdarn_cluster.time_utils import *
times_unique_index = time_days_to_index(times_unique_index)
index_time = time_days_to_index(time_num_days)


# ~~ Fanplots ~~
# Note this will create about 1000 plots for 1 day of data, so it takes a while.
"""
from superdarn_cluster.FanPlot import FanPlot
fan_colors = list(colors)
fan_colors.append((0, 0, 0, 1))
fanplot = FanPlot()
fanplot.plot_all(times_unique_dt, times_unique_index, index_time, beam, gate, labels, fan_colors)
"""

# ~~ Plotting all DBSCAN clusters on RTI plot (scatterplot) ~~
for b in range(16):
    fig = plt.figure(figsize=(16,8))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)
        beam_mask = (beam == b)

        xy = X[class_member_mask & core_samples_mask & beam_mask]
        plt.plot(xy[:, 2], xy[:, 1], '.', color=tuple(col), markersize=15)

        xy = X[class_member_mask & ~core_samples_mask & beam_mask]
        plt.plot(xy[:, 2], xy[:, 1], '.', color=tuple(col), markersize=15)

    plt.xlim((np.min(integer_time/time_eps), np.max(integer_time/time_eps)))
    plt.ylim((np.min(gate/gate_eps), np.max(gate/gate_eps)))
    plt.title('Beam %d \n Clusters: %d   Eps: %.2f   MinPts: %d ' % (b, n_clusters_, eps, minPts))
    plt.savefig('dbscan beam ' + str(b))
    plt.close()

from superdarn_cluster.utilities import plot_clusters
stats_i = [0, 1, 2, 3, 4, 7, 8]
data_flat_unscaled[:, 2] = np.abs(data_flat_unscaled[:, 2])
data_flat_unscaled[:, 3] = np.abs(data_flat_unscaled[:, 3])
plot_clusters(labels, data_flat_unscaled[:, stats_i], data_flat_unscaled[:, 6], 
               gate, vel, np.array(feature_names)[stats_i], range_max, start_time, end_time, save=True, base_path='dbscan ')

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
    #return med_vel < 33.1 + 0.139 * med_wid - 0.00133 * (med_wid ** 2)
    return med_vel < 30 - med_wid * 1.0 / 3.0
    #return med_vel < 15

stats_i = [0, 1, 2, 3, 4, 7, 8]
gs_flg = np.zeros(len(time_num_days))
for k in unique_labels:
    class_member_mask = (labels == k)
    k_vel = vel[class_member_mask]
    k_wid = wid[class_member_mask]
    if np.sum(class_member_mask) < 500:
        gs_flg[class_member_mask] = get_gs_flg(k_vel, k_wid)
        continue
    
    data = gmm_data[class_member_mask]
    estimator = BayesianGaussianMixture(n_components=3,
                                        covariance_type='full', max_iter=500,
                                        random_state=0, n_init=5, init_params='kmeans',
                                        weight_concentration_prior=1,
                                        weight_concentration_prior_type='dirichlet_process')
    estimator.fit(data)
    clust_labels = estimator.predict(data)
    
    class_label_i = np.where(class_member_mask)[0]
    for cl in np.unique(clust_labels):
        cluster_mask = clust_labels == cl
        #plt.scatter(data[cluster_mask, 2], data[cluster_mask, 3])
        #plt.show()
        clust_label_mask = np.zeros(len(time_num_days), dtype=bool)
        clust_label_mask[class_label_i[cluster_mask]] = True
        gs_flg[class_member_mask & clust_label_mask] = get_gs_flg(k_vel[cluster_mask], k_wid[cluster_mask])
    
    data_for_stats = data_flat_unscaled[class_member_mask]
    data_for_stats = data_for_stats[:, stats_i]
    clust_time = time_num_days[class_member_mask]
    clust_gate = gate[class_member_mask]
    clust_vel = vel[class_member_mask]
    names_for_stats = np.array(feature_names)[stats_i]
    
    plot_clusters(clust_labels, data_for_stats, clust_time, clust_gate, clust_vel, names_for_stats, range_max, start_time, end_time, 
                  save=True, base_path='gmm dbscan cluster ' + str(k) + " ")
    plt.close()


# ~~ IS/GS Colormesh RTI ~~
from superdarn_cluster.utilities import plot_is_gs_colormesh

fig = plt.figure(figsize=(16, 4))
ax = plt.subplot(111)
time_num_days_unique = [date2num(d) for d in data_dict['datetime']]

plot_is_gs_colormesh(ax, time_num_days_unique, time_num_days, gate, gs_flg, range_max, plot_indeterminate=False)
plt.title('gs is colormesh code threshold BoxCox.png')
plt.savefig('gs is colormesh code threshold BoxCox.png')