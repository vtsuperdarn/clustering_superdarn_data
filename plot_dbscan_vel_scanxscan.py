from superdarn_cluster.STDBSCAN_fast import STDBSCAN, dict_to_csr_sparse
import os
import numpy as np
import time
from superdarn_cluster.utilities import plot_is_gs_colormesh

# TODO combine into plot_fanplots_dbscan & delete

""" Change this to specify what experiment you are running """
dir = '../experiments/7-25-18 DBSCAN + Ratio Vel (scan x scan)/'
clust_dir = 'cluster fanplots/'
vel_dir = 'velocity fanplots/'           # It is not necessary to generate fanplots for each of these similar scripts - they will all be the same.
if not os.path.exists(dir):
    os.makedirs(dir)
if not os.path.exists(dir + clust_dir):
    os.makedirs(dir + clust_dir)
if not os.path.exists(dir + vel_dir):
    os.makedirs(dir + vel_dir)

gs_threshold = 'code'

stats = open(dir+'0_stats.txt', 'w')

""" Get data """
import pickle
rad_date = "sas_2018-02-07"
ngate = 75
nbeam = 16
data_dict = pickle.load(open("./pickles/%s_scans.pickle" % rad_date, 'rb'))     # Note: this data_dict is scan by scan, other data_dicts may be beam by beam

""" Classification methods """
def blanchard_gs_flg(v, w, type='code'):
    med_vel = np.median(np.abs(v))
    med_wid = np.median(np.abs(w))
    if type == 'paper':
        return med_vel < 33.1 + 0.139 * med_wid - 0.00133 * (med_wid ** 2)    # Found in 2009 paper
    if type == 'code':
        return med_vel < 30 - med_wid * 1.0 / 3.0                             # Found in RST code

gs_label = []

""" Experiment: Running scan-by-scan velocity DB and classifying output """
# from scipy.stats import boxcox
scans_to_use = range(len(data_dict['gate']))
values = [np.abs(data_dict['vel'][i]) for i in scans_to_use]
# The scaling trick to get multiple DBSCAN eps does not work with my _fast versions of DBSCAN, because they are
# still based on searching in a grid-based manner - look at beam +- eps, gate +- eps to find points

data, data_i = dict_to_csr_sparse(data_dict, ngate, nbeam, values)
from superdarn_cluster.FanPlot import FanPlot
import matplotlib.pyplot as plt

eps1 = 2.0
min_pts = 5
eps2, d_eps = 2.0, 2.0      #  this is relative difference for ratio
gdb = STDBSCAN(eps1, eps2, d_eps, min_pts, ngate, nbeam)

dt = 0
vel = data_dict['vel']
for i in scans_to_use:
    """ Run STDB """

    t0 = time.time()
    labels = gdb.fit([data[i]], [data_i[i]])        # hacky fix
    dt += time.time() - t0

    unique_clusters = np.unique(np.hstack(labels))
    stats.write('Vel DBSCAN Clusters: '+ str(unique_clusters) + '\n')
    cluster_colors = list(
        plt.cm.jet(
            np.linspace(0, 1, len(unique_clusters) + 1)))  # one extra unused color at index 0 (no cluster label == 0)
    # randomly re-arrange colors for contrast in adjacent clusters
    np.random.seed(0)
    np.random.shuffle(cluster_colors)

    cluster_colors.append((0, 0, 0, 1))  # black for noise

    """ Cluster fanplot """
    clusters = np.unique(labels[0])
    # Plot a fanplot
    fanplot = FanPlot(nrange=ngate, nbeam=nbeam)
    for c in clusters:
        label_mask = labels[0] == c
        fanplot.plot(data_dict['beam'][i][label_mask], data_dict['gate'][i][label_mask], cluster_colors[c])
    plt.title('Vel DBSCAN fanplot\neps1 = %.2f    eps2 = %.2f    deps = %.2f    minPts = %d'
              % (eps1, eps2, d_eps, min_pts))
    filename = '%s_1eps%.2f_2eps%.2f_deps%.2f_mpts%d_scan%d_fanplot.png' % (rad_date, eps1, eps2, d_eps, min_pts, i)
    # plt.show()
    plt.savefig(dir + clust_dir + filename)
    plt.close()

    """ Velocity fanplot """
    # Plot velocity fanplot
    fanplot = FanPlot(nrange=ngate, nbeam=nbeam)
    # BoxCox velcoity scaling
    #vel_step = 0.1
    #vel_ranges = list(np.linspace(-4, 4, 21))
    #vel_ranges.insert(0, -20)
    #vel_ranges.append(20)
    # Regular velocity scaling
    vel_step = 5
    vel_ranges = list(range(-50, 51, vel_step))
    vel_ranges.insert(0, -9999)
    vel_ranges.append(9999)
    cmap = plt.cm.jet       # use 'viridis' to make this redgreen colorblind proof
    vel_colors = cmap(np.linspace(0, 1, len(vel_ranges)))
    for s in range(len(vel_ranges) - 1):
        step_mask = (vel[i] >= vel_ranges[s]) & (vel[i] <= (vel_ranges[s+1]))
        fanplot.plot(data_dict['beam'][i][step_mask], data_dict['gate'][i][step_mask], vel_colors[s])
    filename = 'vel_scan%d_fanplot.png' % (i)
    fanplot.add_colorbar(vel_ranges, cmap)
    #plt.show()
    plt.savefig(dir + vel_dir + filename)
    plt.close()


    labels = labels[0]  # hacky fix

    """ Assign GS/IS labels 
    Here, the Ribiero method does not make sense, because none of the clusters have a duration - its 1 scan at a time.
    """
    gs_labels_i = np.array([-1]*len(labels))
    for c in unique_clusters:
        # Skip noise, leave it labelled as -1
        if c == -1:
            continue
        label_mask = labels == c
        gs_flg = blanchard_gs_flg(data_dict['vel'][i][label_mask], data_dict['wid'][i][label_mask], gs_threshold)
        gs_labels_i[label_mask] = gs_flg
    gs_label.append(gs_labels_i)

stats.write('Time elapsed: %.2f s\n' % dt)
stats.close()

""" Plot IS/GS on RTI plot """
gs_label = np.hstack(gs_label)              #TODO maybe have label written to the data_dict, then it can save a pickle as output
time_flat = np.hstack(np.array(data_dict['time'])[scans_to_use])
unique_time = np.unique(time_flat)
beams = np.hstack(np.array(data_dict['beam'])[scans_to_use])
gates = np.hstack(np.array(data_dict['gate'])[scans_to_use])

for b in range(nbeam):
    fig = plt.figure(figsize=(16, 4))
    ax = plt.subplot(111)
    beam_filter = b == beams
    plot_is_gs_colormesh(ax, unique_time, time_flat[beam_filter], gates[beam_filter], gs_label[beam_filter], ngate,
                         plot_indeterminate=True)
    name = 'gs is colormesh %s threshold beam %d' % (gs_threshold, b)
    plt.title(name)
    plt.savefig(dir + name + '.png')


