from superdarn_cluster.utilities import plot_is_gs_colormesh
import matplotlib.pyplot as plt
import pickle
import datetime as dt
import os
import numpy as np

rad = 'cvw'
dates = [(2018, 2, 7), (2017, 5, 30), (2017, 8, 20), (2017, 10, 16), (2017, 12, 19), (2018, 4, 5)]
pickle_alg = 'GBDBSCAN (scan x scan)'
gs_threshold = 'code'
pickle_dir = './pickles/%s/' % pickle_alg
plot_rti = True
graph_dir = '../experiments/7-27-18 %s/' % pickle_alg
if not os.path.exists(graph_dir):
    os.makedirs(graph_dir)

# DBSCAN PDFs
fig0 = plt.figure(num=0, figsize=(17, 9))
ax0 = plt.subplot(221)
ax0.set_title('IS day by day')
ax1 = plt.subplot(222)
ax1.set_title('GS day by day')
ax2 = plt.subplot(223)
ax2.set_title('IS total')
ax3 = plt.subplot(224)
ax3.set_title('GS total')
fig0.suptitle('IS/GS distributions\n%s\n%s threshold' % (pickle_alg, gs_threshold))

# Traditional PDFs
fig1 = plt.figure(num=1, figsize=(17, 9))
ax4 = plt.subplot(221)
ax4.set_title('IS day by day')
ax5 = plt.subplot(222)
ax5.set_title('GS day by day')
ax6 = plt.subplot(223)
ax6.set_title('IS total')
ax7 = plt.subplot(224)
ax7.set_title('GS total')
fig1.suptitle('IS/GS distributions\ntraditional code threshold')

is_lim = 100
gs_lim = 50

is_bins = list(range(0, 1000, 5))
is_bins.append(9999)                # noise bin
gs_bins = list(range(0, 500, 5))
gs_bins.append(9999)                # noise bin

combined_vel = []
combined_gsflg = []
combined_trad_gsflg = []

def plot_pdf(ax, data, bins, label=None, line='-'):
    y, binEdges = np.histogram(data, bins=bins)
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    y = y / len(data)
    ax.plot(bincenters, y, line, label=label)

for date in dates:
    plt.figure(0)       # DBSCAN
    year, month, day = date[0], date[1], date[2]
    start_time = dt.datetime(year, month, day)
    date_str = '%d-%02d-%02d' % (year, month, day)
    picklefile = '%s_%s_labels.pickle' % (rad, date_str)
    data_dict = pickle.load(open(pickle_dir + picklefile, 'rb'))

    vel = np.abs(np.hstack(data_dict['vel']))
    gs_flg = np.hstack(data_dict['gs_flg'])
    combined_vel.extend(vel)
    combined_gsflg.extend(gs_flg)

    """ Plot DBSCAN """
    trad_gs_mask = gs_flg == 1           # Can be 0, 1, -1
    trad_is_mask = gs_flg == 0
    plt.figure(0)
    plot_pdf(ax0, vel[trad_is_mask], is_bins, date_str)
    plot_pdf(ax1, vel[trad_gs_mask], gs_bins, date_str)

    """ Plot traditional """
    trad_gs_flg = np.hstack(data_dict['trad_gsflg'])
    combined_trad_gsflg.extend(trad_gs_flg)
    trad_gs_mask = trad_gs_flg == 1           # Can be 0, 1, -1
    trad_is_mask = trad_gs_flg == 0

    plt.figure(1)
    plot_pdf(ax4, vel[trad_is_mask], is_bins, date_str)
    plot_pdf(ax5, vel[trad_gs_mask], gs_bins, date_str)

    if plot_rti:
        """ Plot IS/GS on RTI plot """
        time_flat = np.hstack(np.array(data_dict['time']))
        unique_time = np.unique(time_flat)
        beams = np.hstack(np.array(data_dict['beam']))
        gates = np.hstack(np.array(data_dict['gate']))
        ngate = data_dict['nrang']
        nbeam = int(np.max(beams)) + 1

        if not os.path.exists(graph_dir+date_str):
            os.makedirs(graph_dir+date_str)

        for b in range(nbeam):
            fig1 = plt.figure(num=2, figsize=(16, 4))       # Create a new figure for RTI plot
            ax = plt.subplot(111)
            beam_filter = b == beams
            plot_is_gs_colormesh(ax, unique_time, time_flat[beam_filter], gates[beam_filter], gs_flg[beam_filter], ngate,
                                 plot_indeterminate=True)
            name = '%s %s gs is colormesh %s threshold beam %d' % (rad.upper(), date_str, gs_threshold, b)
            plt.title(name)
            plt.savefig(graph_dir + date_str + '/' + name + '.png')
            plt.close(2)


""" Save algorithm results """
plt.figure(0)
ax0.legend()
ax1.legend()
combined_vel = np.array(combined_vel)
combined_gsflg = np.array(combined_gsflg)
combined_trad_gsflg = np.array(combined_trad_gsflg)
ax4.legend()
ax5.legend()
trad_is_mask = combined_trad_gsflg == 0
trad_gs_mask = combined_trad_gsflg == 1

is_mask = combined_gsflg == 0
gs_mask = combined_gsflg == 1
plot_pdf(ax2, combined_vel[is_mask], is_bins, label='DBSCAN')
plot_pdf(ax3, combined_vel[gs_mask], gs_bins, label='DBSCAN')

plot_pdf(ax2, combined_vel[trad_is_mask], is_bins, label='Traditional', line='--')
plot_pdf(ax3, combined_vel[trad_gs_mask], gs_bins, label='Traditional', line='--')

ax2.legend()
ax3.legend()

ax0.set_xlim(left=0, right=1100)
ax1.set_xlim(left=0, right=550)
ax2.set_xlim(left=0, right=1100)
ax2.set_ylim(bottom=0)
ax3.set_xlim(left=0, right=550)
ax3.set_ylim(bottom=0)
plt.savefig('%spdfs_%s_threshold.png' % (graph_dir, gs_threshold))

ax0.set_xlim(left=0, right=is_lim)
ax1.set_xlim(left=0, right=gs_lim)
ax2.set_xlim(left=0, right=is_lim)
ax3.set_xlim(left=0, right=gs_lim)
plt.savefig('%spdfs_%s_threshold_ZOOM.png' % (graph_dir, gs_threshold))


""" Save traditional results"""
plt.figure(1)

plot_pdf(ax6, combined_vel[trad_is_mask], is_bins)
plot_pdf(ax7, combined_vel[trad_gs_mask], gs_bins)

ax4.set_xlim(left=0, right=1100)
ax5.set_xlim(left=0, right=550)
ax6.set_xlim(left=0, right=1100)
ax7.set_xlim(left=0, right=550)
plt.savefig('%spdfs_trad_threshold.png' % (graph_dir))

ax4.set_xlim(left=0, right=is_lim)
ax5.set_xlim(left=0, right=gs_lim)
ax6.set_xlim(left=0, right=is_lim)
ax7.set_xlim(left=0, right=gs_lim)
plt.savefig('%spdfs_trad_threshold_ZOOM.png' % (graph_dir))
