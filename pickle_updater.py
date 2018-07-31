""" Update a pickle with IS/GS and cluster labels from one of the algorithms """

import os
import numpy as np
import time
import datetime as dt
import pickle
from superdarn_cluster.utilities import ribiero_gs_flg, blanchard_gs_flg


def gbdbscan_timefilter(data_dict, stats, params, gs_threshold='code'):
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
    f = params['f']
    g = params['g']
    pts_ratio = params['pts_ratio']

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
        if gs_threshold == 'Ribiero':
            labels_i = ribiero_gs_flg(vel_flat[cluster_mask_flat], time_flat[cluster_mask_flat])
        elif gs_threshold == 'code' or gs_threshold == 'paper':
            labels_i = blanchard_gs_flg(vel_flat[cluster_mask_flat], wid_flat[cluster_mask_flat], gs_threshold)
        else:
            raise ('Bad gs_threshold: ' + gs_threshold)
        gs_label[cluster_mask_flat] = labels_i
        stats.write('%d: velocity var %.2f      width var %.2f\n' % (
                    c, np.var(np.abs(vel_flat[cluster_mask_flat])), np.var(np.abs(wid_flat[cluster_mask_flat]))))
        stats.write('    velocity mean %.2f      width mean %.2f\n' % (
                    np.mean(np.abs(vel_flat[cluster_mask_flat])), np.mean(np.abs(wid_flat[cluster_mask_flat]))))

    # Convert gs flags and cluster labels to scan-by-scan
    gs_flg = []
    i = 0
    for s in data_dict['vel']:
        gs_flg.append(gs_label[i:i + len(s)])
        i += len(s)
    return gs_flg, labels


def gbdbscan(data_dict, stats, params, gs_threshold='code'):
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
    f = params['f']
    g = params['g']
    pts_ratio = params['pts_ratio']
    gdb = GridBasedDBSCAN(f, g, pts_ratio, ngate, nbeam, dr, dtheta, r_init)

    gs_flgs = []
    labels = []
    dt = 0
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
            if gs_threshold == 'Ribiero':
                raise ('Bad gs_threshold for scan x scan:' + gs_threshold)

            gs_flg = blanchard_gs_flg(data_dict['vel'][i][label_mask], data_dict['wid'][i][label_mask], gs_threshold)
            gs_flgs_i[label_mask] = gs_flg
        gs_flgs.append(gs_flgs_i)
    return gs_flgs, labels


""" Customize these params """
algs = ['DBSCAN', 'GBDBSCAN', 'DBSCAN + Vel', 'GBDBSCAN + Vel']
timefilter = False
alg_i = 1
gs_threshold = 'code'
exper_dir = '../experiments'
pickle_dir = './pickles'
rad = 'sas'
alg_dir = '%s + %s (%s)' % (algs[alg_i], gs_threshold, 'timefilter' if timefilter else 'scan x scan')
dates = [(2018, 2, 7), (2017, 5, 30), (2017, 8, 20), (2017, 10, 16), (2017, 12, 19), (2018, 2, 7), (2018, 4, 5)]
params = {'f': 0.2, 'g': 1, 'pts_ratio': 0.6}       # timefilter GBDB
#params = {'f': 0.3, 'g': 2, 'pts_ratio': 0.3}       # scanxscan GBDB

dir = pickle_dir + '/' + alg_dir
if not os.path.exists(dir):
    os.makedirs(dir)
stats = open(dir + '/' + '0_stats.txt', 'w')

""" Get data """
for date in dates:
    year, month, day = date[0], date[1], date[2]
    start_time = dt.datetime(year, month, day)
    date_str = '%d-%02d-%02d' % (year, month, day)
    data_dict = pickle.load(open("%s/%s_%s_scans.pickle" % (pickle_dir, rad, date_str), 'rb'))     # Note: this data_dict is scan by scan, other data_dicts may be beam by beam

    stats.write('~~~~  %s %s  ~~~~\n' % (rad, date_str))
    stats.write('Generated on %s\n' % dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    stats.flush()   # Flush the buffer so contents show up in file

    gs_flg, clust_flg = gbdbscan_timefilter(data_dict, stats, params, gs_threshold=gs_threshold)
    data_dict['gs_flg'] = gs_flg
    data_dict['clust_flg'] = clust_flg
    data_dict['params'] = params

    output = '%s_%s_labels.pickle' % (rad, date_str)
    pickle.dump(data_dict, open(dir + '/' + output, 'wb'))

stats.close()
