import numpy as np
from sklearn.cluster import DBSCAN
import copy
from superdarn_cluster.time_utils import time_days_to_index
from superdarn_cluster.utilities import blanchard_gs_flg
from sklearn.mixture import GaussianMixture
from superdarn_cluster.utilities import plot_clusters
from scipy.stats import boxcox


class DBSCAN_GMM:

    def __init__(self, params, alg='GMM'):
        self.time_eps = params['time_eps']
        self.beam_eps = params['beam_eps']
        self.gate_eps = params['gate_eps']
        self.eps = params['eps']
        self.min_pts = params['min_pts']
        self.n_clusters = params['n_clusters']
        self.alg = alg

    def fit(self, beam, gate, time, vel, wid):
        """ Do DBSCAN """
        time_integ = time_days_to_index(time)  # integration time, float
        X = np.column_stack((beam / self.beam_eps, gate / self.gate_eps, time_integ / self.time_eps))
        db = DBSCAN(eps=self.eps, min_samples=self.min_pts).fit(X)
        labels = db.labels_
        db_labels_unique = np.unique(labels)
        print('DBSCAN clusters: '+str(np.max(db_labels_unique)))

        """ Do GMM """
        # ~~ BoxCox on velocity and width
        bx_vel, h_vel = boxcox(np.abs(vel))
        bx_wid, h_wid = boxcox(np.abs(wid))
        gmm_data = np.column_stack((bx_vel, bx_wid, time_integ, gate))

        for dbc in db_labels_unique:
            db_cluster_mask = (labels == dbc)
            if dbc == -1:
                continue
            num_pts = np.sum(db_cluster_mask)
            # Sometimes DBSCAN will find tiny clusters due to this:
            # https://stackoverflow.com/questions/21994584/can-the-dbscan-algorithm-create-a-cluster-with-less-than-minpts
            # I don't want to keep these clusters, so label them as noise
            if num_pts < self.min_pts:
                labels[db_cluster_mask] = -1
            # TODO base this on variance(abs(whatev, maybe vel, wid, range gate, power)))
            if num_pts < 500:
                continue
            if self.alg == 'GMM':

                data = gmm_data[db_cluster_mask]
                # Using 3 components will hopefully separate it into 3 groups: IS, GS, and Noise (high variance)
                # Or maybe it will create 2 IS clusters and 1 GS
                # It's not perfect, but BayesGMM doesn't behave well at higher number of clusters.
                # Perhaps the number of clusters used should actually depend on the size of the dataset
                estimator = GaussianMixture(n_components=self.n_clusters+int(num_pts/2500), #self.n_clusters,
                                covariance_type='full', max_iter=500,
                                random_state=0, n_init=5, init_params='kmeans')
                estimator.fit(data)
                gmm_labels = estimator.predict(data)
                gmm_labels += np.max(labels) + 1
                labels[db_cluster_mask] = gmm_labels    # TODO make sure this works okay
            elif self.alg == 'k-means':
                raise('k-means not yet implememnted')
            else:
                raise('bad alg '+str(self.alg))

        return labels
