import sys
sys.path.insert(0,'..')
from algorithms.algorithm import GMMAlgorithm
import numpy as np

class GMM(GMMAlgorithm):
    """
    GMM for SuperDARN data.
    """
    def __init__(self, start_time, end_time, rad,
                 n_clusters=10, cov='full',
                 features=['beam', 'gate', 'time', 'vel', 'wid'],
                 BoxCox=False,
                 load_model=False,
                 save_model=False):
        super().__init__(start_time, end_time, rad,
                         {'n_clusters' : n_clusters,
                          'cov': cov,
                          'features': features,
                          'BoxCox': BoxCox},
                         load_model=load_model)

        if not load_model:
            clust_flg, self.runtime = self._gmm(self._get_gmm_data_array())
            self.clust_flg = self._1D_to_scanxscan(clust_flg)
            print(np.unique(np.hstack(self.clust_flg)))
        if save_model:
            self._save_model()


if __name__ == "__main__":
    import datetime
    start_time = datetime.datetime(2018, 2, 7)
    end_time = datetime.datetime(2018, 2, 8)
    gmm = GMM(start_time, end_time, 'cvw', n_clusters=10, load_model=True)
    print(gmm.__dict__.keys())
    gmm.plot_rti(8, 'Blanchard code')
