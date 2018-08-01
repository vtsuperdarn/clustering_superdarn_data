import os
import numpy as np
from superdarn_cluster.utilities import plot_is_gs_colormesh, plot_vel_colormesh, add_colorbar
import pickle
import datetime as dt
import matplotlib.dates as mdates
from superdarn_cluster.FanPlot import FanPlot
import matplotlib.pyplot as plt

# TODO add colormesh individual cluster plots with stats for max usefulness

""" Customize these params """
algs = ['DBSCAN', 'GBDBSCAN', 'DBSCAN + Vel', 'GBDBSCAN + Vel', 'DBSCAN + GMM']
timefilter = True
alg_i = 4
gs_threshold = 'code'
exper_dir = '../experiments'
pickle_dir = './pickles'
yr, mo, day = 2018, 2, 7
rad = 'sas'

alg_dir = '%s + %s (%s)' % (algs[alg_i], gs_threshold, 'timefilter' if timefilter else 'scan x scan')
dir = '%s/%s %s' % (exper_dir, dt.datetime.now().strftime('%m-%d-%Y'), alg_dir)

clust_dir = '%s/cluster fanplots' % (dir)
vel_dir = '%s/velocity fanplots' % (dir) # It is not necessary to generate fanplots for each of these similar scripts - they will all be the same.
rti_dir = '%s/rti/%d/%s' % (dir, yr, rad)
if not os.path.exists(dir):
    os.makedirs(dir)
if not os.path.exists(clust_dir):
    os.makedirs(clust_dir)
if not os.path.exists(vel_dir):
    os.makedirs(vel_dir)
if not os.path.exists(rti_dir):
    os.makedirs(rti_dir)
stats = open(dir + '/0_stats.txt', 'w')
stats.write(alg_dir + '\n')

""" Get data """
rad_date = "%s_%d-%02d-%02d" % (rad, yr, mo, day)
data_dict = pickle.load(open('%s/%s/%s_labels.pickle' % (pickle_dir, alg_dir, rad_date), 'rb'))
scans_to_use = list(range(len(data_dict['gate'])))

labels = data_dict['clust_flg']
unique_clusters = np.unique(np.hstack(labels))
stats.write('Clusters: %s\n' % str(unique_clusters))
cluster_colors = list(
    plt.cm.jet(
        np.linspace(0, 1, len(unique_clusters) + 1)))  # one extra unused color at index 0 (no cluster label == 0)
# randomly re-arrange colors for contrast in adjacent clusters
np.random.seed(0)  # always produce same cluster colors on subsequent runs
np.random.shuffle(cluster_colors)

vel = data_dict['vel']
time = data_dict['time']
ngate = int(data_dict['nrang'])
nbeam = int(data_dict['nbeam'])

""" Optional (SLOW) loop: produce fanplots color-coded by cluster and by velocity """
for i in scans_to_use:
    clusters = np.unique(labels[i])
    # Plot a fanplot
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

    plt.title('%s fanplot\nparams: %s' % (alg_dir, data_dict['params']))
    filename = '%s/%s_scan%d_fanplot.png' % (clust_dir, rad_date, i)
    # plt.show()
    plt.savefig(filename)
    plt.close()
    """ Velocity map """
    # Plot velocity fanplot
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


""" Plot individual clusters with stats """
# TODO

""" Plot IS/GS on RTI plot """
gs_label = np.hstack(data_dict['gs_flg'])
time_flat = np.hstack(np.array(data_dict['time'])[scans_to_use])
unique_time = np.unique(time_flat)
beams = np.hstack(np.array(data_dict['beam'])[scans_to_use])
gates = np.hstack(np.array(data_dict['gate'])[scans_to_use])
vels = np.hstack(np.array(data_dict['vel'])[scans_to_use])

date_str = dt.datetime(yr, mo, day).strftime('%Y-%m-%d')
hours = mdates.HourLocator(byhour=range(0, 24, 4))

for b in range(nbeam):
    fig = plt.figure(figsize=(14, 10))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    beam_filter = b == beams
    plot_is_gs_colormesh(ax1, unique_time, time_flat[beam_filter], gates[beam_filter], gs_label[beam_filter], ngate,
                         plot_indeterminate=True, plot_closerange=True)
    name = '%s %s                  IS/GS                  %s / %s threshold                  beam %d' \
                % (rad.upper(), date_str, algs[alg_i], gs_threshold, b)
    ax1.set_title(name)
    ax1.xaxis.set_major_locator(hours)
    plot_vel_colormesh(fig, ax2, unique_time, time_flat[beam_filter], gates[beam_filter], vels[beam_filter], ngate)
    name = '%s %s                  Velocity                  %s / %s threshold                  beam %d' \
                % (rad.upper(), date_str, algs[alg_i], gs_threshold, b)
    ax2.set_title(name)
    ax2.xaxis.set_major_locator(hours)
    plt.savefig('%s/%s_%d%02d%02d_%02d.jpg' % (rti_dir, rad, yr, mo, day, b))
    fig.clf()           # Necessary to prevent memory explosion
    plt.close()



