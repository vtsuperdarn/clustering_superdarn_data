""" Update a pickle with IS/GS and cluster labels from one of the algorithms """

import os
import numpy as np
import time
import datetime as dt
import pickle
from superdarn_cluster.utilities import ribiero_gs_flg, blanchard_gs_flg
from sklearn.mixture import GaussianMixture


def update_flags(data_dict, stats, clust_labels):
    vel = np.hstack(np.abs(data_dict['vel']))
    t = np.hstack(data_dict['time'])
    wid = np.hstack(np.abs(data_dict['wid']))
    rib_labels = np.zeros(len(clust_labels))
    code_labels = np.zeros(len(clust_labels))
    paper_labels = np.zeros(len(clust_labels))

    for c in np.unique(clust_labels):
        clust_mask = c == clust_labels
        if c == -1:
            rib_labels[clust_mask] = -1  # Noise flag
            # TODO the others also need to update this!!!!!!!!!!!
        else:
            rib_labels[clust_mask] = ribiero_gs_flg(vel[clust_mask], t[clust_mask])
            code_labels[clust_mask] = blanchard_gs_flg(vel[clust_mask], wid[clust_mask], 'code')
            paper_labels[clust_mask] = blanchard_gs_flg(vel[clust_mask], wid[clust_mask], 'paper')
        stats.write('%d: velocity var %.2f      width var %.2f\n' % (
            c, np.var(np.abs(vel[clust_mask])), np.var(np.abs(wid[clust_mask]))))
        stats.write('    velocity mean %.2f      width mean %.2f\n' % (
            np.mean(np.abs(vel[clust_mask])), np.mean(np.abs(wid[clust_mask]))))

    # Make the GS/cluster labels scan-by-scan
    rib_flg = []
    code_flg = []
    paper_flg = []
    clust_flg = []
    i = 0
    for s in data_dict['vel']:
        rib_flg.append(rib_labels[i:i + len(s)])
        code_flg.append(code_labels[i:i + len(s)])
        paper_flg.append(paper_labels[i:i + len(s)])
        clust_flg.append(clust_labels[i:i + len(s)])
        i += len(s)
    data_dict['clust_flg'] = clust_flg
    data_dict['ribiero_flg'] = rib_flg
    data_dict['paper_flg'] = paper_flg
    data_dict['code_flg'] = code_flg


def gmm(data_dict, data, stats, params):
    n_clusters = params['n_clusters']
    cov = params['cov']
    estimator = GaussianMixture(n_components=n_clusters,
                    covariance_type=cov, max_iter=500,
                    random_state=0, n_init=5, init_params='kmeans')
    t0 = time.time()
    estimator.fit(data)
    dt = time.time() - t0
    clust_labels = estimator.predict(data)
    update_flags(data_dict, stats, clust_labels)

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

    update_flags(data_dict, stats, clust_labels)


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

    update_flags(data_dict, stats, np.hstack(labels))

def gbdbscan_timefilter_gmm(data_dict, stats, params):
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
    unique_labels = np.unique(labels)
    for c in unique_labels:
        clust_mask = c == unique_labels
        n_pts = np.sum(clust_mask)
        if n_pts < 500:
            continue
        estimator = GaussianMixture(n_components=params['n_clusters'],
                                    covariance_type='full', max_iter=500,
                                    random_state=0, n_init=5, init_params='kmeans')
        gmm_labels = estimator.predict(data[clust_mask])
        gmm_labels += np.max(labels) + 1
        labels[clust_mask] = gmm_labels

    dt = time.time() - t0
    stats.write('Time elapsed: %.2f s\n' % dt)
    unique_clusters = np.unique(np.hstack(labels))
    stats.write('Grid-based DBSCAN Clusters: %s\n' % str(unique_clusters))

    update_flags(data_dict, stats, np.hstack(labels))

# TODO add flags/params to data_dict
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
    algs = ['DBSCAN', 'DBSCAN + GMM', 'GMM', 'GBDBSCAN (scan x scan)', 'GBDBSCAN (timefitler)', 'GBDBSCAN (timefitler) + GMM']
    alg_i = 2
    exper_dir = '../experiments'
    pickle_dir = './pickles'
    rad = 'sas'
    alg_dir = algs[alg_i]
    dates = [(2017, 10, 16), (2018, 2, 7), (2017, 5, 30)] #[(2018, 2, 7), (2017, 5, 30), (2017, 8, 20), (2017, 10, 16), (2017, 12, 19), (2018, 2, 7), (2018, 4, 5)]

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
        if algs[alg_i] == 'GBDBSCAN':
            if timefilter:
                params = {'f': 0.2, 'g': 1, 'pts_ratio': 0.6}  # timefilter GBDB
                gbdbscan_timefilter(data_dict, stats, params)
            else:      # scan x scan
                params = {'f': 0.3, 'g': 2, 'pts_ratio': 0.3}       # scanxscan GBDB
                gbdbscan(data_dict, stats, params)
        elif algs[alg_i] == 'DBSCAN + GMM':
            params = {'time_eps': 20.0, 'beam_eps': 3.0, 'gate_eps': 1.0, 'eps': 1.0, 'min_pts': 5, 'n_clusters': 3}
            dbscan_gmm(data_dict, stats, params)
        elif algs[alg_i] == 'GMM':
            # Features for GMM to use
            from scipy.stats import boxcox
            data = np.column_stack((np.hstack(data_dict['time']),
                                    np.hstack(data_dict['gate']),
                                    np.hstack(data_dict['beam']),
                                    boxcox(np.abs(np.hstack(data_dict['vel'])))[0],
                                    boxcox(np.abs(np.hstack(data_dict['wid'])))[0]))
            params = {'n_clusters': 30, 'cov': 'full'}
            gmm(data_dict, data, stats, params)
        elif algs[alg_i] == 'GBDBSCAN (timefilter) + GMM':
            params = {'f': 0.2, 'g': 1, 'pts_ratio': 0.6, 'n_clusters': 30, 'cov': 'full'}  # timefilter GBDB
            gbdbscan_timefilter_gmm(data_dict, stats, params)
        else:
            raise('Bad alg')

        data_dict['params'] = params
        output = '%s_%s_labels.pickle' % (rad, date_str)
        pickle.dump(data_dict, open(dir + '/' + output, 'wb'))

    stats.close()
