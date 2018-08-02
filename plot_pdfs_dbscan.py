from superdarn_cluster.utilities import plot_is_gs_colormesh
import matplotlib.pyplot as plt
import pickle
import datetime as dt
import os
import numpy as np

# TODO combine the 3 plot_pdf scripts into 2 script that takes a pickle file? does it make sense to pickle pca? lol

rad = 'sas'
dates = [(2017, 5, 30), (2017, 8, 20), (2017, 10, 16), (2017, 12, 19), (2018, 2, 7), (2018, 4, 5)]
#pickle_alg = 'GBDBSCAN + Ribiero (timefilter)'
pickle_alg = 'GBDBSCAN (scan x scan)'
gs_threshold = 'code'
pickle_dir = './pickles/%s/' % pickle_alg
plot_rti = False
graph_dir = '../experiments/7-30-18 %s/' % pickle_alg
if not os.path.exists(graph_dir):
    os.makedirs(graph_dir)

# DBSCAN PDFs
fig0 = plt.figure(num=0, figsize=(17, 10))
ax0 = []
ax0.append(plt.subplot(221))
ax0[0].set_title('IS day by day')
ax0.append(plt.subplot(222))
ax0[1].set_title('GS day by day')
ax0.append(plt.subplot(223))
ax0[2].set_title('IS total')
ax0.append(plt.subplot(224))
ax0[3].set_title('GS total')
fig0.suptitle('%s IS/GS distributions\n%s\n%s threshold' % (rad.upper(), pickle_alg, gs_threshold))
for ax in ax0:
    ax.set_xlabel('|Velocity| (m/s)')
    ax.set_ylabel('Probability')

"""
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
fig1.suptitle('%s IS/GS distributions\ntraditional code threshold' % (rad.upper()))
"""

# Virtual height difference histogram
fig2 = plt.figure(num=2, figsize=(17, 9))
ax8 = plt.subplot(221)
ax8.set_title('IS day by day')
ax9 = plt.subplot(222)
ax9.set_title('GS day by day')
ax10 = plt.subplot(223)
ax10.set_title('IS / GS total')
fig2.suptitle('%s Virtual height difference (h*-h)/(h*)\n%s threshold' % (rad.upper(), gs_threshold))


is_lim = 100
gs_lim = 50

is_bins = list(range(0, 1000, 5))
is_bins.append(9999)                # noise bin
gs_bins = list(range(0, 500, 5))
gs_bins.append(9999)                # noise bin

combined_vel = []
combined_gsflg = []
combined_trad_gsflg = []
combined_vheight_diff = []

def plot_pdf(ax, data, bins, label=None, line='-', hist=False):
    y, binEdges = np.histogram(data, bins=bins)
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    if not hist:
        y = y / len(data)                       # Creates a PDF from a histogram
    ax.plot(bincenters, y, line, label=label)

def probable_virtual_height(r):

    h_star = np.zeros(len(r))
    half_E_mask = r < 790
    half_F_mask = (r >= 790) & (r < 2130)
    one_half_F_mask = r >= 2130
    h_star[half_E_mask] = 108.974 + 0.0191271 * r[half_E_mask] + 6.68283e-5 * r[half_E_mask]**2
    h_star[half_F_mask] = 384.416 - 0.178640 * r[half_F_mask] + 1.81405e-4 * r[half_F_mask]**2
    h_star[one_half_F_mask] = 1098.28 - 0.354557 * r[one_half_F_mask] + 9.39961e-5 * r[one_half_F_mask]**2

    #h_star = 384.416 - 0.178640 * r + 1.81405e-4 * r**2
    return h_star

def get_range(gate):
    return 180 + 45*gate

for date in dates:
    plt.figure(0)       # DBSCAN
    year, month, day = date[0], date[1], date[2]
    start_time = dt.datetime(year, month, day)
    date_str = '%d-%02d-%02d' % (year, month, day)
    picklefile = '%s_%s_labels.pickle' % (rad, date_str)
    data_dict = pickle.load(open(pickle_dir + picklefile, 'rb'))

    vel = np.abs(np.hstack(data_dict['vel']))
    trad_gs_flg = np.hstack(data_dict['trad_gsflg'])
    if gs_threshold == 'trad':
        gs_flg = trad_gs_flg
    else:
        gs_flg = np.hstack(data_dict['gs_flg'])
    combined_vel.extend(vel)
    combined_gsflg.extend(gs_flg)
    combined_trad_gsflg.extend(trad_gs_flg)

    """ Plot DBSCAN """
    gs_mask = gs_flg == 1           # Can be 0, 1, -1
    is_mask = gs_flg == 0
    plt.figure(0)
    plot_pdf(ax0[0], vel[is_mask], is_bins, date_str)
    plot_pdf(ax0[1], vel[gs_mask], gs_bins, date_str)

    """ Plot virtual height difference - see Blanchard 2009 s2.4, Chisham 2008 s3.1"""
    # h is virtual height calculated from elevation angle and range
    gate = np.hstack(data_dict['gate'])
    elv = np.hstack(data_dict['elv'])
    r = get_range(gate)
    e = elv * np.pi / 180
    h = r * np.sin(e)
    # h* is predicted virtual height for various regions of the ionosphere
    h_star = probable_virtual_height(r)

    plt.figure(2)
    h_diff = np.abs((h_star - h) / h_star)
    combined_vheight_diff.extend(h_diff)
    plot_pdf(ax8, h_diff[is_mask], 100, label=date_str)
    plot_pdf(ax9, h_diff[gs_mask], 100, label=date_str)


    if plot_rti:
        """ Plot IS/GS on RTI plot """
        time_flat = np.hstack(np.array(data_dict['time']))
        unique_time = np.unique(time_flat)
        beams = np.hstack(np.array(data_dict['beam']))
        gates = np.hstack(np.array(data_dict['gate']))
        ngate = data_dict['nrang']
        nbeam = int(np.max(beams)) + 1

        rti_dir = graph_dir + rad + '/' + date_str + '/'
        if not os.path.exists(rti_dir):
            os.makedirs(rti_dir)

        for b in range(nbeam):
            figx = plt.figure(num=3, figsize=(16, 4))       # Create a new figure for RTI plot
            ax = plt.subplot(111)
            beam_filter = b == beams
            plot_is_gs_colormesh(ax, unique_time, time_flat[beam_filter], gates[beam_filter], gs_flg[beam_filter], ngate,
                                 plot_indeterminate=True)
            name = '%s %s gs is colormesh %s threshold beam %d' % (rad.upper(), date_str, gs_threshold, b)
            plt.title(name)
            plt.savefig(rti_dir + name + '.png')
            #plt.savefig(graph_dir + date_str + '/' + name + '.png')
            plt.close(3)


""" Plot DBSCAN combined """
plt.figure(0)
ax0[0].legend()
ax0[1].legend()
combined_vel = np.array(combined_vel)
combined_gsflg = np.array(combined_gsflg)
combined_trad_gsflg = np.array(combined_trad_gsflg)
trad_is_mask = combined_trad_gsflg == 0
trad_gs_mask = combined_trad_gsflg == 1

is_mask = combined_gsflg == 0
gs_mask = combined_gsflg == 1
plot_pdf(ax0[2], combined_vel[is_mask], is_bins, label='DBSCAN')
plot_pdf(ax0[3], combined_vel[gs_mask], gs_bins, label='DBSCAN')

# Plot traditional for comparison
plot_pdf(ax0[2], combined_vel[trad_is_mask], is_bins, label='Traditional', line='--')
plot_pdf(ax0[3], combined_vel[trad_gs_mask], gs_bins, label='Traditional', line='--')

ax0[2].legend()
ax0[3].legend()

for i, ax in enumerate(ax0):
    ax.set_ylim(bottom=0)
    if i % 2 == 0: ax.set_xlim(left=0, right=1100)
    else: ax.set_xlim(left=0, right=550)
plt.savefig('%s%s_pdfs_%s_threshold.png' % (graph_dir, rad, gs_threshold))

# Zoom in and create another image
for i, ax in enumerate(ax0):
    ax.set_ylim(bottom=0)
    if i % 2 == 0: ax.set_xlim(left=0, right=is_lim)
    else: ax.set_xlim(left=0, right=gs_lim)
plt.savefig('%s%s_pdfs_%s_threshold_ZOOM.png' % (graph_dir, rad, gs_threshold))

""" Plot virtual height difference combined """
plt.figure(2)
combined_vheight_diff = np.array(combined_vheight_diff)
plot_pdf(ax10, combined_vheight_diff[is_mask], 100, label='IS')
plot_pdf(ax10, combined_vheight_diff[gs_mask], 100, label='GS')
ax8.legend()
ax9.legend()
ax10.legend()
plt.savefig('%s%s_vheight_diff_pdf_%s_threshold.png' % (graph_dir, rad, gs_threshold))