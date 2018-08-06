from algorithms.Algorithm import GMMAlgorithm
import numpy as np

class GMM(GMMAlgorithm):
    """
    GMM for SuperDARN data.
    """
    def __init__(self, start_time, end_time, rad,
                 n_clusters=10, cov='full',
                 features=['beam', 'gate', 'time', 'vel', 'wid'],
                 BoxCox=False,
                 useSavedResult=False):
        super().__init__(start_time, end_time, rad,
                         {'n_clusters' : n_clusters,
                          'cov': cov,
                          'features': features,
                          'BoxCox': BoxCox},
                         useSavedResult=useSavedResult)

        if not useSavedResult:
            clust_flg, self.runtime = self._gmm(self._get_gmm_data_array())
            self.clust_flg = self._1D_to_scanxscan(clust_flg)
            print(np.unique(np.hstack(self.clust_flg)))


if __name__ == "__main__":
    import datetime
    start_time = datetime.datetime(2018, 2, 7)
    end_time = datetime.datetime(2018, 2, 8)
    gmm = GMM(start_time, end_time, 'cvw', n_clusters=10, useSavedResult=True)
    #gmm.save_result()
    print(gmm.__dict__.keys())
    gmm.plot_rti(8, 'Blanchard code')
