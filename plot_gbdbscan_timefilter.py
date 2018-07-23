from superdarn_cluster.GridBasedDBSCAN_timefilter_fast import GridBasedDBSCAN, dict_to_csr_sparse
import os
import numpy as np
from superdarn_cluster.utilities import plot_is_gs_colormesh

# Want to do this with a few different options:
# With the time DBSCAN, without the time DBSCAN
#
# Various classification method - Blanchard, AJ's, 15 m/s
# Keep track of everything, and just work your way through it slowly. It's just plug and chug.

""" Change this to specify what experiment you are running """
dir = '7-23-18 timefilter GBDBSCAN GS IS flags/'
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

def ribiero_gs_flg(cluster):
    #TODO calculate duration, and find the high/low ratio of velocity
    # then use those to find GS/IS flag
    # consider using Angeline's methodology for this
    pass

""" Experiment 1: Running time-filter GBDB and classifying output """
""" Grid-based DBSCAN """
from superdarn_cluster.FanPlot import FanPlot
import matplotlib.pyplot as plt


scan_i = 0            # ~1400 scans/day for SAS and CVW


# Good way to tune this one: Use only the first 100 scans of SAS 2-7-18, because they tend to get all clustered together with bad params,
# but then there is a long period of quiet so there's time separation between that big cluster and everything else.

scans_to_use = list(range(len(data_dict['gate'])))
values = [[True]*len(data_dict['gate'][i]) for i in scans_to_use]
data, data_i = dict_to_csr_sparse(data_dict, ngate, nbeam, values)

dr = 45
dtheta = 3.3
r_init = 180
f = 0.2
g = 1
pts_ratio = 0.6
gdb = GridBasedDBSCAN(f, g, pts_ratio, ngate, nbeam, dr, dtheta, r_init)
import time

t0 = time.time()
labels = gdb.fit(data, data_i)
dt = time.time() - t0
print('Time elapsed: %.2f s' % dt)

unique_clusters = np.unique(np.hstack(labels))
print('Grid-based DBSCAN Clusters: ', unique_clusters)
cluster_colors = list(
    plt.cm.jet(
        np.linspace(0, 1, len(unique_clusters) + 1)))  # one extra unused color at index 0 (no cluster label == 0)
# randomly re-arrange colors for contrast in adjacent clusters
np.random.seed(0)                   # always produce same cluster colors on subsequent runs
np.random.shuffle(cluster_colors)

cluster_colors.append((0, 0, 0, 1))  # black for noise
vel = data_dict['vel']

""" Optional (SLOW) loop: produce fanplots color-coded by cluster and by velocity """
for i in scans_to_use:
    clusters = np.unique(labels[i])
    # Plot a fanplot
    fanplot = FanPlot(nrange=ngate, nbeam=nbeam)
    for c in clusters:
        label_mask = labels[i] == c
        beam_c = data_dict['beam'][i][label_mask]
        gate_c = data_dict['gate'][i][label_mask]
        fanplot.plot(beam_c, gate_c, cluster_colors[c])
        if c != -1:
            m = int(len(beam_c) / 2)                          # Beam is sorted, so this is roughly the index of the median beam
            fanplot.text(str(c), beam_c[m], gate_c[m])        # Label each cluster with its cluster #
    plt.title('Grid-based DBSCAN fanplot\nf = %.2f    g = %d    pts_ratio = %.2f' % (f, g, pts_ratio))
    filename = dir + '%s_f%.2f_g%d_ptRatio%.2f_scan%d_fanplot.png' % (rad_date, f, g, pts_ratio, i)
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

    filename = dir + 'vel_scan%d_fanplot.png' % (i)
    fanplot.add_colorbar(vel_ranges, cmap)
    # plt.show()
    plt.savefig(filename)
    plt.close()


""" Plot clusters on RTI plot """
# TODO

""" Get GS/IS labels for each cluster """
vel_flat = np.hstack(np.array(data_dict['vel'])[scans_to_use])
wid_flat = np.hstack(np.array(data_dict['wid'])[scans_to_use])
gs_label = np.array([-1] * len(vel_flat))             # Initialize labels to -1
gs_threshold = 'code'

print('~~~ Cluster Stats ~~~')
for c in unique_clusters:
    if c == -1:
        continue        # Skip the noise cluster - no need to set values since gs_label is intialized to -1
    cluster_mask = [labels[i] == c for i in range(len(labels))]     # List of 1-D arrays, scan by scan
    cluster_mask_flat = np.hstack(cluster_mask)                     # Single 1-D array
    labels_i = blanchard_gs_flg(vel_flat[cluster_mask_flat], wid_flat[cluster_mask_flat], gs_threshold)
    gs_label[cluster_mask_flat] = labels_i
    # TODO fix this so its absolute value
    print('%d: velocity var %.2f      width var %.2f' % (c, np.var(vel_flat[cluster_mask_flat]), np.var(wid_flat[cluster_mask_flat])))
    print('    velocity mean %.2f      width mean %.2f' % (np.mean(vel_flat[cluster_mask_flat]), np.mean(wid_flat[cluster_mask_flat])))
    # TODO maybe apply GMM on clusters with variance above a certain threshold, ie vel>10000 &or/ wid>1000


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


