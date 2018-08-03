from algorithms.Algorithm import Algorithm
import numpy as np
import time
from sklearn.mixture import GaussianMixture

class GMM(Algorithm):
    """
    GMM for SuperDARN data.
    """

    def _get_data_array(self):
        data = []
        for feature in self.params['features']:
            vals = self.data_dict[feature]
            data.append(np.hstack(vals))
        return np.column_stack(data)


    def _gmm(self):
        data = self._get_data_array()
        n_clusters = self.params['n_clusters']
        cov = self.params['cov']
        estimator = GaussianMixture(n_components=n_clusters,
                                    covariance_type=cov, max_iter=500,
                                    random_state=0, n_init=5, init_params='kmeans')
        t0 = time.time()
        estimator.fit(data)
        runtime = time.time() - t0
        clust_flg = estimator.predict(data) + 1
        return self._1D_to_scanxscan(clust_flg), runtime


    def __init__(self, start_time, end_time, rad,
                 n_clusters=10, cov='full',
                 features=['beam', 'gate', 'time', 'vel', 'wid'],
                 loadPickle=False):
        super().__init__(start_time, end_time, rad,
                         {'n_clusters' : n_clusters,
                          'cov': cov,
                          'features': features},
                         loadPickle=loadPickle)

        if not loadPickle:
            self.clust_flg, self.runtime = self._gmm()
            print(np.unique(np.hstack(self.clust_flg)))

if __name__ == "__main__":
    import datetime
    start_time = datetime.datetime(2018, 2, 7)
    end_time = datetime.datetime(2018, 2, 8)

    gmm = GMM(start_time, end_time, 'cvw')
    gmm.pickle()
    print(gmm.__dict__.keys())
    gmm.plot_rti(8, 'Ribiero')
