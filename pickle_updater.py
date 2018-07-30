""" Update a pickle with IS/GS and cluster labels from one of the algorithms """

import os
import numpy as np
import time
import datetime as dt
import pickle


#TODO some of these functions could be simplified, and all should be moved at some point

""" Classification methods """
def blanchard_gs_flg(v, w, type='code'):
    med_vel = np.median(np.abs(v))
    med_wid = np.median(np.abs(w))
    if type == 'paper':
        return med_vel < 33.1 + 0.139 * med_wid - 0.00133 * (med_wid ** 2)  # Found in 2009 paper
    if type == 'code':
        return med_vel < 30 - med_wid * 1.0 / 3.0  # Found in RST code


# TODO width is not used
def ribiero_gs_flg(v, w, t):
    L = np.abs(t[-1] - t[0]) * 24
    high = np.sum(np.abs(v) > 15.0)
    low = np.sum(np.abs(v) <= 15.0)
    if low == 0:
        R = 1.0  # TODO hmm... this works right?
    else:
        R = high / low  # High vel / low vel ratio
    # See Figure 4 in Ribiero 2011
    if L > 14.0:
        return True  # GS
    elif L > 3:
        if R > 0.2:
            return False  # IS
        else:
            return True
    elif L > 2:
        if R > 0.33:
            return False
        else:
            return True
    elif L > 1:
        if R > 0.475:
            return False
        else:
            return True
    # Addition by Burrell 2018 "Solar influences..."
    else:
        if R > 0.5:
            return False
        else:
            return True
    # Classic Ribiero 2011
    # else:
    #    return False


def gbdbscan_timefilter(data_dict, stats, gs_threshold='code'):
    from superdarn_cluster.GridBasedDBSCAN_timefilter_fast import GridBasedDBSCAN, dict_to_csr_sparse

    scans_to_use = list(range(len(data_dict['gate'])))
    values = [[True] * len(data_dict['gate'][i]) for i in scans_to_use]
    ngate = int(data_dict['nrang'])
    nbeam = int(data_dict['nbeam'])
    data, data_i = dict_to_csr_sparse(data_dict, values)

    """ Set up GBDBSCAN (change params here, they are hardcoded for now) """
    dr = 45
    dtheta = 3.3
    r_init = 180
    f = 0.2
    g = 1
    pts_ratio = 0.6
    gdb = GridBasedDBSCAN(f, g, pts_ratio, ngate, nbeam, dr, dtheta, r_init)
    t0 = time.time()
    labels = gdb.fit(data, data_i)
    dt = time.time() - t0
    stats.write('Time elapsed: %.2f s\n' % dt)
    unique_clusters = np.unique(np.hstack(labels))
    stats.write('Grid-based DBSCAN Clusters: %s\n' % str(unique_clusters))

    """ Get GS/IS labels for each cluster """
    vel_flat = np.hstack(np.array(data_dict['vel'])[scans_to_use])
    wid_flat = np.hstack(np.array(data_dict['wid'])[scans_to_use])
    time_flat = np.hstack(np.array(data_dict['time'])[scans_to_use])
    gs_label = np.array([-1] * len(vel_flat))  # Initialize labels to -1
    for c in unique_clusters:
        if c == -1:
            continue  # Skip the noise cluster - no need to set values since gs_label is intialized to -1
        cluster_mask = [labels[i] == c for i in range(len(labels))]  # List of 1-D arrays, scan by scan
        cluster_mask_flat = np.hstack(cluster_mask)  # Single 1-D array
        # TODO have this handle the gs_threshold to choose classification method
        labels_i = ribiero_gs_flg(vel_flat[cluster_mask_flat], wid_flat[cluster_mask_flat],
                                  time_flat[cluster_mask_flat])
        gs_label[cluster_mask_flat] = labels_i

    # Convert gs flags and cluster labels to scan-by-scan
    gs_flg = []
    i = 0
    for s in data_dict['vel']:
        gs_flg.append(gs_label[i:i + len(s)])
        i += len(s)
    return gs_flg, labels


def gbdbscan(data_dict, stats, gs_threshold='code'):
    from superdarn_cluster.GridBasedDBSCAN_timefilter_fast import GridBasedDBSCAN, dict_to_csr_sparse
    scans_to_use = range(len(data_dict['gate']))
    values = [[True] * len(data_dict['gate'][i]) for i in scans_to_use]
    # The scaling trick to get multiple DBSCAN eps does not work with my _fast versions of DBSCAN, because they are
    # still based on searching in a grid-based manner - look at beam +- eps, gate +- eps to find points
    ngate = int(data_dict['nrang'])
    nbeam = int(data_dict['nbeam'])
    data, data_i = dict_to_csr_sparse(data_dict, values)

    dr = 45
    dtheta = 3.3
    r_init = 180
    f = 0.2
    g = 1
    pts_ratio = 0.6
    gdb = GridBasedDBSCAN(f, g, pts_ratio, ngate, nbeam, dr, dtheta, r_init)
    dt = 0

    gs_flgs = []
    labels = []
    for i in scans_to_use:
        """ Run GBDB """
        t0 = time.time()
        clust_labels = gdb.fit([data[i]], [data_i[i]])[0]  # hacky fix to get timefilter GBDB to look like scan-by-scan
        labels.append(clust_labels)
        dt += time.time() - t0
        unique_clusters = np.unique(np.hstack(clust_labels))
        stats.write('Vel DBSCAN Clusters: ' + str(unique_clusters) + '\n')

        """ Assign GS/IS labels 
        Here, the Ribiero method does not make sense, because none of the clusters have a duration - its 1 scan at a time.
        """
        gs_flgs_i = np.array([-1] * len(clust_labels))
        for c in unique_clusters:
            # Skip noise, leave it labelled as -1
            if c == -1:
                continue
            label_mask = clust_labels == c
            gs_flg = blanchard_gs_flg(data_dict['vel'][i][label_mask], data_dict['wid'][i][label_mask], gs_threshold)
            gs_flgs_i[label_mask] = gs_flg
        gs_flgs.append(gs_flgs_i)
    return gs_flgs, labels


rad = 'sas'
dates = [(2018, 2, 7), (2017, 5, 30), (2017, 8, 20), (2017, 10, 16), (2017, 12, 19), (2018, 2, 7), (2018, 4, 5)]
alg = 'GBDBSCAN + Ribiero (timefilter)'
dir = './pickles/%s/' % alg

gs_threshold = 'AJ'
if not os.path.exists(dir):
    os.makedirs(dir)
stats = open(dir + '0_stats.txt', 'w')

""" Get data """
for date in dates:
    year, month, day = date[0], date[1], date[2]
    start_time = dt.datetime(year, month, day)
    date_str = '%d-%02d-%02d' % (year, month, day)
    data_dict = pickle.load(open("./pickles/%s_%s_scans.pickle" % (rad, date_str), 'rb'))     # Note: this data_dict is scan by scan, other data_dicts may be beam by beam

    stats.write('~~~~  %s %s  ~~~~\n' % (rad, date_str))
    stats.write('Generated on %s\n' % dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    gs_flg, clust_flg = gbdbscan_timefilter(data_dict, stats, gs_threshold=gs_threshold)
    data_dict['gs_flg'] = gs_flg
    data_dict['clust_flg'] = clust_flg

    output = '%s_%s_labels.pickle' % (rad, date_str)
    pickle.dump(data_dict, open(dir + output, 'wb'))

stats.close()
