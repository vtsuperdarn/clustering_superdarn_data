import sys
sys.path.insert(0,'..')
from algorithms.algorithm import GMMAlgorithm
import numpy as np
from sklearn.cluster import DBSCAN
import time

class DBSCAN_GMM(GMMAlgorithm):
    """
    Run DBSCAN on space/time features, then GMM on space/time/vel/wid
    """
    def __init__(self, start_time, end_time, rad,
                 beam_eps=3, gate_eps=1, scan_eps=1,  # DBSCAN
                 minPts=5, eps=1,  # DBSCAN
                 n_clusters=5, cov='full',  # GMM
                 features=['beam', 'gate', 'time', 'vel', 'wid'],  # GMM
                 BoxCox=False,  # GMM
                 load_model=False,
                 save_model=False):
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
                         load_model=load_model)
        if not load_model:
            clust_flg, self.runtime = self._dbscan_gmm()
            # Randomize flag #'s so that colors on plots are not close to each other
            # (necessary for large # of clusters, but not for small #s)
            self.clust_flg = self._1D_to_scanxscan(clust_flg)
            print('DBSCAN+GMM clusters: ' + str(np.max(clust_flg)))
        if save_model:
            self._save_model()


    def _dbscan_gmm(self):
        # Run DBSCAN on space/time features
        X = self._get_dbscan_data_array()
        t0 = time.time()
        db = DBSCAN(eps=self.params['eps'],
                    min_samples=self.params['min_pts']
                    ).fit(X)
        db_runtime = time.time() - t0
        # Print # of clusters created by DBSCAN
        db_flg = db.labels_
        gmm_data = self._get_gmm_data_array()
        clust_flg, gmm_runtime = self._gmm_on_existing_clusters(gmm_data, db_flg)
        return clust_flg, db_runtime + gmm_runtime


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



import sys

if __name__ == '__main__':

    import datetime
    from datetime import datetime as dt

    dates = [dt(2017, 1, 17), dt(2017, 3, 13), dt(2017, 4, 4), dt(2017, 5, 30), dt(2017, 8, 20),
             dt(2017, 9, 20), dt(2017, 10, 16), dt(2017, 11, 14), dt(2017, 12, 8), dt(2017, 12, 17),
             dt(2017, 12, 18), dt(2017, 12, 19), dt(2018, 1, 25), dt(2018, 2, 7), dt(2018, 2, 8),
             dt(2018, 3, 8), dt(2018, 4, 5)]
    rad = sys.argv[1]

    if rad == 'sas':
        threshold = 'Blanchard code'
        vel_max=200
        vel_step=25
    elif rad == 'cvw':
        threshold = 'Ribiero'
        vel_max=100
        vel_step=10
    else:
        print('Cant use that radar')
        exit()

    print(rad)
    print(dates)
    print(threshold)

    for date in dates:
        start_time = date
        end_time = date + datetime.timedelta(days=1)
        dbgmm = DBSCAN_GMM(start_time, end_time, rad,
                           load_model=False, save_model=True, BoxCox=True)
        dbgmm.plot_rti('*', threshold, vel_max=100, vel_step=10, show_fig=False, save_fig=True)
        #dbgmm.plot_fanplots(start_time, end_time, vel_max=100, vel_step=10, show=False, save=True)
