from superdarn_cluster.GridBasedDBSCAN_fast import GridBasedDBSCAN, dict_to_csr_sparse
import os
import numpy as np
from superdarn_cluster.utilities import plot_is_gs_colormesh


""" Change this to specify what experiment you are running """
dir = '7-22-18 scan by scan STDBSCAN GS IS flags/'
if not os.path.exists(dir):
    os.makedirs(dir)

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

""" Experiment 1: Running scan-by-scan STDB and classifying output """
scans_to_use = range(len(data_dict['vel']))
gs_threshold = 'code'
data, data_i = dict_to_csr_sparse(data_dict, ngate, nbeam, scans_to_use)

from superdarn_cluster.FanPlot import FanPlot
import matplotlib.pyplot as plt

# Solid params across the board: f=0.3, g=2
dr = 45
dtheta = 3.3
r_init = 180
f = 0.3
g = 2
eps2 = 0.5
d_eps = 0.5
pts_ratio = 0.3
gdb = GridBasedDBSCAN(f, g, pts_ratio, ngate, nbeam, dr, dtheta, r_init)
import time

t = 0
vel = data_dict['vel']
gs_label = []
""" Run GBDBSCAN scan by scan and determine IS/GS label """
for i in scans_to_use:
    t0 = time.time()
    labels = gdb.fit(data[i], data_i[i])
    dt = time.time() - t0
    t += dt

    unique_clusters = np.unique(labels)

    print('Grid-based DBSCAN Clusters: ', unique_clusters)
    cluster_colors = list(
        plt.cm.plasma(
            np.linspace(0, 1, len(unique_clusters)+1)))  # one extra unused color at index 0 (no cluster label == 0)
    cluster_colors.append((0, 0, 0, 1))  # black for noise

    """ Cluster Fanplot """
    fanplot = FanPlot(nrange=ngate, nbeam=nbeam)
    for c in unique_clusters:
        label_mask = labels == c
        fanplot.plot(data_dict['beam'][i][label_mask], data_dict['gate'][i][label_mask], cluster_colors[c])
    plt.title('Grid-based DBSCAN fanplot\nf = %.2f    g = %d    pts_ratio = %.2f' % (f, g, pts_ratio))
    filename = '%s_f%.2f_g%d_ptRatio%.2f_scan%d_fanplot.png' % (rad_date, f, g, pts_ratio, i)
    # plt.show()
    plt.savefig(dir + filename)
    plt.close()

    """ Velocity Fanplot """
    # Plot velocity fanplot
    fanplot = FanPlot(nrange=ngate, nbeam=nbeam)
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
    plt.savefig(dir + filename)
    plt.close()


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



print('Time elapsed: %.2f s' % t)


