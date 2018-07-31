from sklearn.cluster import DBSCAN
import os
import numpy as np
import time
from superdarn_cluster.utilities import plot_is_gs_colormesh

""" 
Note about results : This will produce some clusters with size < min_pts
See the post here for an explanation: https://github.com/scikit-learn/scikit-learn/issues/5031
"""

""" Change this to specify what experiment you are running """
dir = '../experiments/7-30-18 scan by scan DBSCAN GS IS flags/'
clust_dir = 'cluster fanplots/'
vel_dir = 'velocity fanplots/'           # It is not necessary to generate fanplots for each of these similar scripts - they will all be the same.
if not os.path.exists(dir):
    os.makedirs(dir)
if not os.path.exists(dir + clust_dir):
    os.makedirs(dir + clust_dir)
if not os.path.exists(dir + vel_dir):
    os.makedirs(dir + vel_dir)

gs_threshold = 'code'

stats = open(dir + '0_stats.txt', 'w')

""" Get data """
import pickle
yr, mo, day = 2018, 2, 7
rad = 'sas'
rti_dir = 'rti/%d/%s/' % (yr, rad)
rad_date = "sas_%d-%02d-%02d" % (yr, mo, day)
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

""" Experiment: Running scan-by-scan DBSCAN and classifying output """
scans_to_use = range(len(data_dict['gate']))
beam = np.array(data_dict['beam'])#[scans_to_use]
gate = np.array(data_dict['gate'])#[scans_to_use]
vel = np.array(data_dict['vel'])#[scans_to_use]

# To give DBSCAN a fighting chance, should "use 2 epsilons" (ie: prescale the data)
beam_eps = 3.0
gate_eps = 1.0
db_eps = 1
min_pts = 5

data = [np.column_stack((beam[i] / beam_eps, gate[i] / gate_eps)) for i in range(len(beam))]

from superdarn_cluster.FanPlot import FanPlot
import matplotlib.pyplot as plt

dt = 0
db = DBSCAN(eps=db_eps, min_samples=min_pts)


for i in scans_to_use:
    """ Run DB """
    t0 = time.time()
    labels = db.fit(data[i]).labels_
    dt += time.time() - t0

    unique_clusters = np.unique(np.hstack(labels))
    stats.write('DBSCAN Clusters: ' + str(unique_clusters) + '\n')
    cluster_colors = list(
        plt.cm.jet(
            np.linspace(0, 1, len(unique_clusters) + 1)))  # one extra unused color at index 0 (no cluster label == 0)
    # randomly re-arrange colors for contrast in adjacent clusters
    np.random.seed(0)
    np.random.shuffle(cluster_colors)

    cluster_colors.append((0, 0, 0, 1))  # black for noise

    """ Cluster fanplot 
    clusters = np.unique(labels)
    # Plot a fanplot
    fanplot = FanPlot(nrange=ngate, nbeam=nbeam)
    for c in clusters:
        label_mask = labels == c
        fanplot.plot(data_dict['beam'][i][label_mask], data_dict['gate'][i][label_mask], cluster_colors[c])
    plt.title('Vel DBSCAN fanplot\nbeam / %.2f    gate / %.2f    eps = %.2f    minPts = %d' % (beam_eps, gate_eps, db_eps, min_pts))
    filename = '%s_beam%.2f_gate%.2f_eps%.2f_mpts%d_scan%d_fanplot.png' % (rad_date, beam_eps, gate_eps, db_eps, min_pts, i)
    # plt.show()
    plt.savefig(dir + clust_dir + filename)
    plt.close()
    """

    """ Velocity fanplot 
    # Plot velocity fanplot
    fanplot = FanPlot(nrange=ngate, nbeam=nbeam)
    vel_step = 25
    vel_ranges = list(range(-200, 201, vel_step))
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
    """

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

rti_dir = dir + 'rti/' + str(yr) + '/' + rad + '/'
if not os.path.exists(rti_dir):
    os.makedirs(rti_dir)

import datetime as dt
date_str = dt.datetime(yr, mo, day).strftime('%Y-%m-%d')

import matplotlib.dates as mdates
hours = mdates.HourLocator(byhour=range(0, 24, 4))

for b in range(nbeam):
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(111)
    beam_filter = b == beams
    plot_is_gs_colormesh(ax, unique_time, time_flat[beam_filter], gates[beam_filter], gs_label[beam_filter], ngate,
                         plot_indeterminate=True, plot_closerange=True)
    name =  '%s %s                  DBSCAN / %s threshold                  beam %d' % (rad.upper(), date_str, gs_threshold, b)
    ax.xaxis.set_major_locator(hours)
    plt.title(name)
    #plt.savefig(dir + name + '.jpg')
    plt.savefig('%s%s_%d%02d%02d_%02d.jpg' % (rti_dir, rad, yr, mo, day, b))

