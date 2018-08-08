import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from algorithms.Algorithm import GBDBAlgorithm, GMMAlgorithm


class GBDBSCAN_GMM(GBDBAlgorithm, GMMAlgorithm):

    def __init__(self, start_time, end_time, rad,
                 f=0.2, g=1, pts_ratio=0.3,                             # GBDB
                 dr=45, dtheta=3.24, r_init=180,                        # GBDB
                 scan_eps=1,                                            # GBDB Timefilter
                 n_clusters=3, cov='full',                              # GMM
                 features=['beam', 'gate', 'time', 'vel', 'wid'],       # GMM
                 BoxCox=False,                                          # GMM
                 useSavedResult=False):
        super().__init__(start_time, end_time, rad,
                         {'f': f,                   # GBDBSCAN stuff
                          'g': g,
                          'pts_ratio': pts_ratio,
                          'scan_eps': scan_eps,     # for scan by scan set = 0, for timefilter set >= 1
                          'dr': dr,
                          'dtheta': dtheta,
                          'r_init': r_init,
                          'features': features,
                          'n_clusters': n_clusters, # GMM stuff
                          'cov': cov,
                          'BoxCox': BoxCox
                          },
                         useSavedResult=useSavedResult)
        if not useSavedResult:
            clust_flg, self.runtime = self._gbdb_gmm()
            self.clust_flg = self._1D_to_scanxscan(clust_flg)


    def _gbdb_gmm(self):
        data, data_i = self._get_gbdb_data_matrix(self.data_dict)
        gbdb_flg, gbdb_runtime = self._gbdb(data, data_i)
        gmm_data = self._get_gmm_data_array()
        clust_flg, gmm_runtime = self._gmm_on_existing_clusters(gmm_data, gbdb_flg)
        return clust_flg, gbdb_runtime + gmm_runtime


if __name__ == '__main__':
    import datetime
    start_time = datetime.datetime(2018, 2, 7)
    end_time = datetime.datetime(2018, 2, 8)
    gbdb_gmm = GBDBSCAN_GMM(start_time, end_time, 'cvw', useSavedResult=False)
    gbdb_gmm.save_result()
    print(gbdb_gmm.__dict__.keys())
    print(gbdb_gmm.runtime)
    gbdb_gmm.plot_rti(8, 'Blanchard code')
    end_time = datetime.datetime(2018, 2, 7, 23, 59)
    gbdb_gmm.plot_fanplots(start_time, end_time, show=False, save=True)