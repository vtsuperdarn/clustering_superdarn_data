import os
import numpy as np
from superdarn_cluster.utilities import plot_is_gs_colormesh, plot_vel_colormesh, plot_clusters_colormesh, plot_stats_table
from superdarn_cluster.FanPlot import FanPlot
import pickle
import datetime as dt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


""" Customize these params """
algs = ['DBSCAN', 'DBSCAN + GMM', 'GMM', 'GBDBSCAN (scan x scan)', 'GBDBSCAN (timefitler)',
        'GBDBSCAN (timefitler) + GMM']
alg_i = 2
gs_threshold = 'Ribiero'
exper_dir = '../experiments/Comparing 4 algs/'
pickle_dir = './pickles'
yr, mo, day = 2017, 10, 16
rad = 'sas'

alg_dir = '%s' % (algs[alg_i])
dir = '%s/%s %s + %s' % (exper_dir, dt.datetime.now().strftime('%m-%d-%Y'), alg_dir, gs_threshold)


if not os.path.exists(dir):
    os.makedirs(dir)


stats = open(dir + '/0_stats.txt', 'w')
stats.write(alg_dir + gs_threshold + '\n')

""" Get data """
rad_date = "%s_%d-%02d-%02d" % (rad, yr, mo, day)
data_dict = pickle.load(open('%s/%s/%s_labels.pickle' % (pickle_dir, alg_dir, rad_date), 'rb'))
scans_to_use = list(range(len(data_dict['gate'])))

labels = data_dict['clust_flg']
unique_clusters = np.unique(np.hstack(labels))
stats.write('Clusters: %s\n' % str(unique_clusters))
cluster_colors = list(
    plt.cm.jet(
        np.linspace(0, 1, np.max(unique_clusters) + 1)))  # one extra unused color at index 0 (no cluster label == 0)
# randomly re-arrange colors for contrast in adjacent clusters
np.random.seed(0)  # always produce same cluster colors on subsequent runs
np.random.shuffle(cluster_colors)

time = data_dict['time']
ngate = int(data_dict['nrang'])
nbeam = int(data_dict['nbeam'])



""" Plot IS/GS on RTI plot """
if gs_threshold == 'Ribiero':
    gs_flg = data_dict['ribiero_flg']
if gs_threshold == 'code':
    gs_flg = data_dict['code_flg']
if gs_threshold == 'paper':
    gs_flg = data_dict['paper_flg']

gs_label = np.hstack(gs_flg)
time_flat = np.hstack(np.array(data_dict['time'])[scans_to_use])
unique_time = np.unique(time_flat)
beams = np.hstack(np.array(data_dict['beam'])[scans_to_use])
gates = np.hstack(np.array(data_dict['gate'])[scans_to_use])
vels = np.hstack(np.array(data_dict['vel'])[scans_to_use])
labels_flat = np.hstack(labels)

date_str = dt.datetime(yr, mo, day).strftime('%Y-%m-%d')
hours = mdates.HourLocator(byhour=range(0, 24, 4))

import copy
clust_range = list(range(-1, int(max(unique_clusters))+1))
randomized_labels_unique = copy.deepcopy(clust_range)
np.random.shuffle(randomized_labels_unique)
random_labels_flat = np.zeros(len(labels_flat))
for c in unique_clusters:
    cluster_mask = c == labels_flat
    if c == -1:
        random_labels_flat[cluster_mask] = -1
    else:
        random_labels_flat[cluster_mask] = randomized_labels_unique[c]

rti_dir = '%s/rti/%d/%s' % (dir, yr, rad)
if not os.path.exists(rti_dir):
    os.makedirs(rti_dir)

for b in range(nbeam):
    fig = plt.figure(figsize=(14, 15))
    ax0 = plt.subplot(311)
    ax1 = plt.subplot(312)
    ax2 = plt.subplot(313)
    beam_filter = b == beams

    plot_clusters_colormesh(ax0, unique_time, time_flat[beam_filter], gates[beam_filter], clust_range, random_labels_flat[beam_filter], ngate)
    name = '%s %s                           Clusters                           %s                            beam %d' \
           % (rad.upper(), date_str, algs[alg_i], b)
    ax0.set_title(name)
    ax0.xaxis.set_major_locator(hours)
    plot_is_gs_colormesh(ax1, unique_time, time_flat[beam_filter], gates[beam_filter], gs_label[beam_filter], ngate,
                         plot_indeterminate=True, plot_closerange=True)
    name = '%s %s                  IS/GS                  %s / %s threshold                  beam %d' \
                % (rad.upper(), date_str, algs[alg_i], gs_threshold, b)
    ax1.set_title(name)
    ax1.xaxis.set_major_locator(hours)
    plot_vel_colormesh(fig, ax2, unique_time, time_flat[beam_filter], gates[beam_filter], vels[beam_filter], ngate)
    name = '%s %s                                                    Velocity                                                    beam %d' \
                % (rad.upper(), date_str, b)
    ax2.set_title(name)
    ax2.xaxis.set_major_locator(hours)
    plt.savefig('%s/%s_%d%02d%02d_%02d.jpg' % (rti_dir, rad, yr, mo, day, b))
    fig.clf()           # Necessary to prevent memory explosion
    plt.close()

""" Optional (SLOW) loop: produce fanplots color-coded by cluster and by velocity     """


vel = data_dict['vel']

vel_dir = '%s/velocity fanplots %d-%d-%d' % (dir, yr, mo, day) # It is not necessary to generate fanplots for each of these similar scripts - they will all be the same.
if not os.path.exists(vel_dir):
    os.makedirs(vel_dir)
clust_fan_dir = '%s/cluster fanplots %d-%d-%d' % (dir, yr, mo, day)
if not os.path.exists(clust_fan_dir):
    os.makedirs(clust_fan_dir)

scans_to_use = range(100)
for i in scans_to_use:
    clusters = np.unique(labels[i])
    # Cluster fanplot
    fanplot = FanPlot(nrange=ngate, nbeam=nbeam)
    for ci, c in enumerate(clusters):
        label_mask = labels[i] == c
        beam_c = data_dict['beam'][i][label_mask]
        gate_c = data_dict['gate'][i][label_mask]
        if c == -1:
            color = (0, 0, 0, 1)
        else:
            color = cluster_colors[c]
            m = int(len(beam_c) / 2)                          # Beam is sorted, so this is roughly the index of the median beam
            fanplot.text(str(c), beam_c[m], gate_c[m])        # Label cluster #
        fanplot.plot(beam_c, gate_c, color)

    plt.title('%s + %s fanplot\nparams: %s' % (alg_dir, gs_threshold, data_dict['params']))
    filename = '%s/%s_scan%d_fanplot.png' % (clust_fan_dir, rad_date, i)
    # plt.show()
    plt.savefig(filename)
    plt.close()

"""
    # Velocity map
    fanplot = FanPlot(nrange=ngate, nbeam=nbeam)
    vel_step = 25
    vel_ranges = list(range(-200, 201, vel_step))
    vel_ranges.insert(0, -9999)
    vel_ranges.append(9999)
    cmap = plt.cm.jet  # use 'viridis' to make this redgreen colorblind proof
    vel_colors = cmap(np.linspace(0, 1, len(vel_ranges)))
    for s in range(len(vel_ranges) - 1):
        step_mask = (vel[i] >= vel_ranges[s]) & (vel[i] <= (vel_ranges[s + 1]))
        fanplot.plot(data_dict['beam'][i][step_mask], data_dict['gate'][i][step_mask], vel_colors[s])

    filename = vel_dir + '/vel_scan%d_fanplot.png' % (i)
    fanplot.add_colorbar(vel_ranges, cmap)
    # plt.show()
    plt.savefig(filename)
    plt.close()
    """


""" OPTIONAL / SLOW :Plot individual clusters on RTI with stats - NOTE: for scan x scan algorithms, these will be NONSENSE """
clust_rti_dir = '%s/cluster rtiplots %d-%d-%d' % (dir, yr, mo, day)
if not os.path.exists(clust_rti_dir):
    os.makedirs(clust_rti_dir)

vel_abs_flat = np.hstack(np.abs(data_dict['vel']))
wid_abs_flat = np.hstack(np.abs(data_dict['wid']))
elv_flat = np.hstack(data_dict['elv'])
beam_flat = np.hstack(data_dict['beam'])
gate_flat = np.hstack(data_dict['gate'])
time_flat = np.hstack(data_dict['time'])
labels_flat = np.hstack(labels)
clust_fig = plt.figure(num=1, figsize=(16, 4))
for c in unique_clusters:
    cluster_mask = labels_flat == c
    if np.sum(cluster_mask) < 100:
        continue                    # Skip small clusters, DBSCAN creates a million of them
    if c == -1:
        color = (0, 0, 0, 1)    # black
    else:
        color = cluster_colors[c]
    fig = plt.figure(num=0, figsize=(16, 8))
    ax1 = plt.subplot(211)
    ax1.scatter(time_flat[cluster_mask], gate_flat[cluster_mask], color=color, s=5)
    ax1.set_ylim(bottom=0, top=ngate)
    ax1.set_xlim(left=time_flat[0], right=time_flat[-1])
    ax1.set_title('%s cluster RTI\nnum points: %d\nparams: %s' % (alg_dir, np.sum(cluster_mask), data_dict['params']))
    ax2 = plt.subplot(212)
    stats_data = {'vel': vel_abs_flat[cluster_mask], 'wid': wid_abs_flat[cluster_mask], 'elv': elv_flat[cluster_mask],
                  'beam': beam_flat[cluster_mask], 'gate': gate_flat[cluster_mask]}
    plot_stats_table(ax2, stats_data)
    filename = '%s/%s_cluster%d_rti.png' % (clust_rti_dir, rad_date, c)
    plt.savefig(filename)
    plt.close()
    fig.clf()

filename = '%s/%s_00_allclusters_rti.png' % (clust_rti_dir, rad_date)
plt.savefig(filename)
plt.close()
clust_fig.clf()
