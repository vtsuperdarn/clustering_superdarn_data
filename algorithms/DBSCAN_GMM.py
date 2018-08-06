import numpy as np
from sklearn.cluster import DBSCAN
from algorithms.Algorithm import GMMAlgorithm
import time

class DBSCAN_GMM(GMMAlgorithm):
    """
    Run DBSCAN on space/time features, then GMM on space/time/vel/wid
    """
    def __init__(self, start_time, end_time, rad,
                 beam_eps=3, gate_eps=1, scan_eps=1,                # DBSCAN
                 minPts=5, eps=1,                                   # DBSCAN
                 n_clusters=3, cov='full',                          # GMM
                 features=['beam', 'gate', 'time', 'vel', 'wid'],   # GMM
                 BoxCox=False,                                      # GMM
                 useSavedResult=False):
        super().__init__(start_time, end_time, rad,
                         {'scan_eps' : scan_eps,
                          'beam_eps': beam_eps,
                          'gate_eps': gate_eps,
                          'eps': eps,
                          'min_pts': minPts,
                          'n_clusters' : n_clusters,
                          'cov': cov,
                          'features': features,
                          'BoxCox': BoxCox},
                         useSavedResult=useSavedResult)
        if not useSavedResult:
            clust_flg, self.runtime = self._dbscan_gmm()
            # Randomize flag #'s so that colors on plots are not close to each other
            # (necessary for large # of clusters, but not for small #s)
            rand_clust_flg = self._randomize_flags(clust_flg)
            self.clust_flg = self._1D_to_scanxscan(rand_clust_flg)
            print('DBSCAN+GMM clusters: ' + str(np.max(clust_flg)))


    def _dbscan_gmm(self):
        # Run DBSCAN on space/time features
        X = self._get_dbscan_data_array()
        t0 = time.time()
        db = DBSCAN(eps=self.params['eps'],
                    min_samples=self.params['min_pts']
                    ).fit(X)
        runtime = time.time() - t0
        # Print # of clusters created by DBSCAN
        clust_flg = db.labels_
        db_labels_unique = np.unique(clust_flg)
        print('DBSCAN clusters: '+str(np.max(db_labels_unique)))
        # Run GMM
        gmm_data = self._get_gmm_data_array()
        for dbc in db_labels_unique:
            db_cluster_mask = (clust_flg == dbc)
            num_pts = np.sum(db_cluster_mask)
            # Sometimes DBSCAN will find tiny clusters due to this:
            # https://stackoverflow.com/questions/21994584/can-the-dbscan-algorithm-create-a-cluster-with-less-than-minpts
            # I don't want to keep these clusters, so label them as noise
            if num_pts < self.params['min_pts']:
                clust_flg[db_cluster_mask] = -1
            if num_pts < 500 or dbc == -1: # skip noise and small clusters
                continue
            data = gmm_data[db_cluster_mask]
            gmm_labels, gmm_runtime = self._gmm(data)
            runtime += gmm_runtime
            gmm_labels += np.max(clust_flg) + 1
            clust_flg[db_cluster_mask] = gmm_labels
        return clust_flg, runtime


    def _get_dbscan_data_array(self):
        beam = np.hstack(self.data_dict['beam'])
        gate = np.hstack(self.data_dict['gate'])
        # Get the scan # for each data point
        scan_num = []
        for i, scan in enumerate(self.data_dict['time']):
            scan_num.extend([i]*len(scan))
        scan_num = np.array(scan_num)
        # Divide each feature by its 'epsilon' to create the illusion of DBSCAN having multiple epsilons
        data = np.column_stack((beam / self.params['beam_eps'],
                                gate / self.params['gate_eps'],
                                scan_num / self.params['scan_eps']))
        return data


if __name__ == '__main__':
    import datetime
    start_time = datetime.datetime(2018, 2, 7)
    end_time = datetime.datetime(2018, 2, 8)
    dbgmm = DBSCAN_GMM(start_time, end_time, 'cvw', useSavedResult=True)
    #dbgmm.save_result()
    dbgmm.plot_rti(8, 'Blanchard code', vel_max=100, vel_step=10, save=True)
    start_time = datetime.datetime(2018, 2, 7, 12, 0, 0)
    end_time = datetime.datetime(2018, 2, 7, 12, 5, 0)
    dbgmm.plot_fanplots(start_time, end_time, vel_max=100, vel_step=10, save=True)