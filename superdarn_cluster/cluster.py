import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.decomposition import PCA
import pickle

def empirical(data_dict):
    """ Old method, slate for removal
    Note that the new method preserved the 'indeterminate' data as -1,
    but sometimes it is more appropriate to treat it as IS (0)

    gs_hops = [1.0, 2.0, 3.0]
    is_hops = [0.5, 1.5, 2.5]

    emp_gate = np.hstack(data_dict['gate'])
    emp_time, emp_gsflg = [], []
    emp_num_scatter = data_dict['num_scatter']

    for i in range(len(emp_num_scatter)):
        emp_time.extend(date2num([data_dict['datetime'][i]]*emp_num_scatter[i]))
        for j in range(len(data_dict['hop'][i])):
            if data_dict['hop'][i][j] in is_hops:
                emp_gsflg.append(0)
            elif data_dict['hop'][i][j] in gs_hops:
                emp_gsflg.append(1)

    return np.array(emp_gsflg)
    """
    return np.hstack(data_dict['gsflg'])

def traditional(data_dict):
    return np.abs(np.hstack(data_dict['gsflg']))


def gmm(data_flat, vel, wid,
        num_clusters=6,  vel_threshold=15,
        bayes=False, weight_prior=1,
        cluster_identities=False,
        make_pickle="", use_pickle=""):

    pickle_dir = "./GMMPickles/"
    if use_pickle != "":
        use_picklefile = pickle_dir + use_pickle
        cluster_labels = pickle.load(open(use_picklefile, 'rb'))

    else:
        # source
        # http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html#sphx-glr-auto-examples-mixture-plot-gmm-covariances-py
        if not bayes:
            estimator = GaussianMixture(n_components=num_clusters,
                                        covariance_type='full', max_iter=500,
                                        random_state=0, n_init=5, init_params='kmeans')
        elif bayes:
            # Params to fine tune:
            # Weight concentration prior - determines number of clusters chosen, 0.1 1 100 1000 no big difference.
            # Concentration prior type - determines the distribution type
            # Max iter - maybe this will need increasing
            # Mean precision prior
            # Mean prior - should this be 0 since we centered the data there? seems to work well!!!
            #
            # n-init - does okay even on just 1 initialization, will increasing it be better?
            # num_clusters - 5 seems like way too few, when you look at clusters of data in just space and time.
            #                See what looks reasonable. I think it'd be good to get a real view of the clustering in space and time,
            #                individual clsuters and not just GS / IS.
            estimator = BayesianGaussianMixture(n_components=num_clusters,
                                                covariance_type='full', max_iter=500,
                                                random_state=0, n_init=5, init_params='kmeans',
                                                weight_concentration_prior=weight_prior,
                                                weight_concentration_prior_type='dirichlet_process')


        estimator.fit(data_flat)
        if bayes:
            print('weights for variational bayes')
            print(estimator.weights_)
            print(np.sum(estimator.weights_ > 0.01), 'clusters used (weight > 1%)')
            print('converged?')
            print(estimator.converged_)
            print('number of iterations to converge')
            print(estimator.n_iter_)
            print('lower bound on likelihood')
            print(estimator.lower_bound_)
            print('weight prior')
            print(weight_prior)


        cluster_labels = estimator.predict(data_flat)

        if make_pickle != "":
            make_picklefile = pickle_dir + make_pickle
            pickle.dump(cluster_labels, open(make_picklefile, 'wb'))

    gs_class_gmm = []
    is_class_gmm = []
    median_vels_gmm = np.zeros(num_clusters)
    median_wids_gmm = np.zeros(num_clusters)
    for i in range(num_clusters):
        scatter = cluster_labels == i
        median_vels_gmm[i] = np.median(np.abs(vel[scatter]))
        median_wids_gmm[i] = np.median(wid[scatter])
        #print median_vels_gmm[i]
        # Traditional style
        #GS = np.abs(median_vels_gmm[i]) < (15 + 0.139*np.abs(median_wids_gmm[i]) - 0.00133*(median_wids_gmm[i]**2))
        #print('Median vel', np.abs(median_vels_gmm[i]))
        #print('Comparison', 15 + 0.139 * np.abs(median_wids_gmm[i]) - 0.00133 * (median_wids_gmm[i] ** 2))
        GS = np.abs(median_vels_gmm[i]) < 15

        """
        # AJ style
        hi_lo_ratio = np.sum(vel > 15) / float(np.sum(vel < 15))
        time = data_flat[:,-1]
        #TODO make this hours
        duration_hr = int(np.max(time[cluster_labels == i]) - np.min(time[cluster_labels == i]) // 3600)
        IS = duration_hr < 14 and ((duration_hr > 3 and hi_lo_ratio > 0.2)  \
             or (duration_hr > 2 and hi_lo_ratio > 0.33) or (duration_hr > 1 and duration_hr > 0.475))
        GS = not IS
        """

        if not GS:
            is_class_gmm.append(i)
        else:
            gs_class_gmm.append(i)
    print()
    gs_flg_gmm = []
    for i in cluster_labels:
        if i in gs_class_gmm:
            gs_flg_gmm.append(1)
        elif i in is_class_gmm:
            gs_flg_gmm.append(0)

    if cluster_identities:
        clusters = [np.where(cluster_labels == i)[0] for i in range(num_clusters)]
        return np.array(gs_flg_gmm), clusters, median_vels_gmm, estimator
    else:
        return np.array(gs_flg_gmm)


def kmeans(data_flat, vel, wid, num_clusters=5,  vel_threshold = 10., num_init = 50, pca=False):
    # Do Principal Component Analysis to reduce dimensionality
    if pca:
        pca = PCA()
        pca.fit(data_flat)
        data_flat = pca.transform(data_flat)

    # Fit kmeans++
    Z_kmeans = KMeans(init = 'k-means++', n_clusters = num_clusters, n_init = num_init).fit_predict(data_flat)
    median_vels_kmeans = np.zeros(num_clusters)
    median_wids_kmeans = np.zeros(num_clusters)
    gs_class_kmeans = []
    is_class_kmeans = []

    for i in range(num_clusters):
        # Find median velocity and spectral width
        median_vels_kmeans[i] = np.median(np.abs(vel[Z_kmeans == i]))
        median_wids_kmeans[i] = np.median(wid[Z_kmeans == i])

        #print median_vels_kmeans[i]
        # Split as GS or IS by median velocity
        if median_vels_kmeans[i] > vel_threshold:
            is_class_kmeans.append(i)
        else:
            gs_class_kmeans.append(i)
    # Create a set of data labels to compare to ground truth, where 1 is GS and 0 is IS
    gs_flg_kmeans = []
    for i in Z_kmeans:
        if i in gs_class_kmeans:
            gs_flg_kmeans.append(1)
        elif i in is_class_kmeans:
            gs_flg_kmeans.append(0)

    gs_flg_kmeans = np.array(gs_flg_kmeans)
    return gs_flg_kmeans


