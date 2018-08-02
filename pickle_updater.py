""" Update a pickle with IS/GS and cluster labels from one of the algorithms """

import os
import numpy as np
import time
import datetime as dt
import pickle
from superdarn_cluster.utilities import ribiero_gs_flg, blanchard_gs_flg
from sklearn.mixture import GaussianMixture


def get_gs_flg(data_dict, stats, clust_labels):
    gs_threshold = data_dict['gs_threshold']
    vel = np.hstack(np.abs(data_dict['vel']))
    t = np.hstack(data_dict['time'])
    wid = np.hstack(np.abs(data_dict['wid']))
    gs_labels = np.zeros(len(clust_labels))

    for c in np.unique(clust_labels):
        clust_mask = c == clust_labels
        if c == -1:
            gs_labels[clust_mask] = -1  # Noise flag
        elif gs_threshold == 'Ribiero':
            gs_labels[clust_mask] = ribiero_gs_flg(vel[clust_mask], t[clust_mask])
        elif gs_threshold == 'code' or gs_threshold == 'paper':
            gs_labels[clust_mask] = blanchard_gs_flg(vel[clust_mask], wid[clust_mask], gs_threshold)
        else:
            raise ('Bad gs_threshold: ' + gs_threshold)
        stats.write('%d: velocity var %.2f      width var %.2f\n' % (
            c, np.var(np.abs(vel[clust_mask])), np.var(np.abs(wid[clust_mask]))))
        stats.write('    velocity mean %.2f      width mean %.2f\n' % (
            np.mean(np.abs(vel[clust_mask])), np.mean(np.abs(wid[clust_mask]))))

    # Make the GS/cluster labels scan-by-scan
    gs_flg = []
    clust_flg = []
    i = 0
    for s in data_dict['vel']:
        gs_flg.append(gs_labels[i:i + len(s)])
        clust_flg.append(clust_labels[i:i + len(s)])
        i += len(s)
    return gs_flg, clust_flg


def gmm(data_dict, stats, params):
    n_clusters = params['n_clusters']
    cov = params['cov']
    data = params['data']

    estimator = GaussianMixture(n_components=n_clusters,
                    covariance_type=cov, max_iter=500,
                    random_state=0, n_init=5, init_params='kmeans')
    t0 = time.time()
    estimator.fit(data)
    dt = time.time() - t0
    clust_labels = estimator.predict(data)
    gs_flg, clust_flg = get_gs_flg(data_dict, stats, clust_labels)
    return gs_flg, clust_flg

def dbscan_gmm(data_dict, stats, params, gs_threshold='code'):
    from superdarn_cluster.DBSCAN_GMM import DBSCAN_GMM
    beam = np.hstack(data_dict['beam'])
    gate = np.hstack(data_dict['gate'])
    t = np.hstack(data_dict['time'])
    vel = np.hstack(data_dict['vel'])
    wid = np.hstack(data_dict['wid'])

    db = DBSCAN_GMM(params)
    t0 = time.time()
    clust_labels = db.fit(beam, gate, t, vel, wid)
    dt = time.time() - t0
    stats.write('Time elapsed: %.2f s\n' % dt)

    gs_flg, clust_flg = get_gs_flg(data_dict, stats, clust_labels)
    return gs_flg, clust_flg


def gbdbscan_timefilter(data_dict, stats, params):
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

    gs_flg, clust_flg = get_gs_flg(data_dict, stats, np.hstack(labels))
    return gs_flg, clust_flg


def gbdbscan(data_dict, stats, params):
    from superdarn_cluster.GridBasedDBSCAN_timefilter_fast import GridBasedDBSCAN, dict_to_csr_sparse
    gs_threshold = data_dict['gs_threshold']
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


if __name__ == '__main__':
    """ Customize these params """
    algs = ['DBSCAN', 'GBDBSCAN', 'DBSCAN + GMM', 'GMM']
    timefilter = True
    alg_i = 2
    gs_threshold = 'code'
    exper_dir = '../experiments'
    pickle_dir = './pickles'
    rad = 'sas'
    alg_dir = '%s + %s (%s)' % (algs[alg_i], gs_threshold, 'timefilter' if timefilter else 'scan x scan')
    dates = [(2017, 10, 16), (2018, 2, 7), (2017, 5, 30)]#[(2018, 2, 7), (2017, 5, 30), (2017, 8, 20), (2017, 10, 16), (2017, 12, 19), (2018, 2, 7), (2018, 4, 5)]

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

        """ SET PARAMS: These params have been chosen as good ones for the algorithm, change them if you want to experiment """
        data_dict['gs_threshold'] = gs_threshold
        if algs[alg_i] == 'GBDBSCAN':
            if timefilter:
                params = {'f': 0.2, 'g': 1, 'pts_ratio': 0.6}  # timefilter GBDB
                gs_flg, clust_flg = gbdbscan_timefilter(data_dict, stats, params, gs_threshold=gs_threshold)
            else:      # scan x scan
                params = {'f': 0.3, 'g': 2, 'pts_ratio': 0.3}       # scanxscan GBDB
                gs_flg, clust_flg = gbdbscan(data_dict, stats, params, gs_threshold=gs_threshold)
        elif algs[alg_i] == 'DBSCAN + GMM':
            params = {'time_eps':20.0, 'beam_eps':3.0, 'gate_eps':1.0, 'eps':1.0, 'min_pts':5, 'n_clusters':3}
            gs_flg, clust_flg = dbscan_gmm(data_dict, stats, params, gs_threshold=gs_threshold)
        elif algs[alg_i] == 'GMM':
            # Features for GMM to use
            data = np.column_stack((np.hstack(data_dict['time']), np.hstack(data_dict['gate']), np.hstack(data_dict['beam']),
                                        np.hstack(data_dict['vel']), np.hstack(data_dict['wid'])))
            params = {'n_clusters':30, 'cov': 'full', 'data': data}

        data_dict['clust_flg'] = clust_flg
        data_dict['params'] = params

        output = '%s_%s_labels.pickle' % (rad, date_str)
        pickle.dump(data_dict, open(dir + '/' + output, 'wb'))

    stats.close()
