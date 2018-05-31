from superdarn_cluster.cluster import *
import itertools
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from superdarn_cluster.dbtools import *

""" Plot a bunch of GMMs models, and compare their AIC and BIC to get model of best fit. """
""" It would be difficult to compare kernels like in the GP. Compare covariance matrices instead? """

#TODO There are a lot of problems in this script, like where the * is on the graph (just remove it), plotting different ranges of clusters

def plot_aic_bic(data_dict, date, save=False):
    # Compare different models - number of features, kernel type. Params like num_clusters, etc. Find ones with lowest AIC or BIC.
    # Brute force approach? Try them all? Tune one thing at a time?

    data_flat, data_time = flatten_data(data_dict,  extras=False)

    # define some models...
    num_clusters = range(1, 11)
    vel_threshhold = [10, 15]
    covariance_type = ['full', 'tied', 'diag', 'spherical']

    aics = []
    bics = []
    i = 1
    for vel_th in vel_threshhold:
        for cov in covariance_type:
            clusters = 6
            model = "Model " + str(i) + " | " + str(vel_th) + " m/s threshhold | " + str(cov) + " covariance | " \
                    + str(clusters) + " clusters"
            print(model)
            i += 1

            # Do regular GMM
            estimator1 = GaussianMixture(n_components=clusters,
                                        covariance_type=cov, max_iter=500,
                                        random_state=0, n_init=5, init_params='kmeans')
            estimator1.fit(data_flat)
            aic1 = estimator1.aic(data_flat)
            bic1 = estimator1.bic(data_flat)
            aics.append(aic1)
            bics.append(bic1)

            print("GMM: " + str(aic1) + " AIC, " + str(bic1) + " BIC")

            # Do variational GMM for big num_clusters
            # They don't have AIC and BIC implemented for this, but it looks easy to do so.
            """
            if num_clusters >= 30:
                estimator2 = BayesianGaussianMixture(n_components=clusters,
                                                    covariance_type=cov, max_iter=500,
                                                    random_state=0, n_init=5, init_params='kmeans',
                                                    weight_concentration_prior=1,
                                                    weight_concentration_prior_type='dirichlet_process')
                estimator2.fit(data_flat)
                #aic2 = (-2 * estimator2.score(data_flat) * data_flat.shape[0] + estimator2.n * np.log(data_flat.shape[0]))
                #bic2 = estimator2.bic()
                #print "Variational bayes: " + str(aic2) + " AIC "
            """

    x = range(1, len(aics) + 1)
    plt.plot(x, aics, label='AIC')
    plt.plot(x, bics, label='BIC')
    plt.legend()
    plt.xlabel("Model #")
    plt.ylabel("Relative AIC or BIC")
    plt.savefig(start_time.__str__() + " AIC and BIC.png")
    plt.close()


def select_cov_and_clusters(data_dict):
    """
    Based on
    http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html
    """
    data_flat, data_time = flatten_data(data_dict, extras=False)
    lowest_bic = np.infty
    bic = []
    best_cv = ''
    best_n_components = 0
    n_components_range = range(1,2)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = GaussianMixture(n_components=n_components,
                                  covariance_type=cv_type, max_iter=500, tol=10e-8,
                                  random_state=0, n_init=5, init_params='kmeans')

            gmm.fit(data_flat)
            bic.append(gmm.bic(data_flat))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
                best_cv = cv_type
                best_n_components = n_components
            print('completed model with ', cv_type, ' cov and ', n_components, ' clusters')

    bic = np.array(bic)
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue', 'darkorange'])
    best_model = best_gmm
    bars = []

    # Plot the BIC scores
    spl = plt.subplot(2, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        pos = range(1, len(n_components_range)+1)
        xpos = np.array(pos) + .2 * (i - 2)

        bars.append(plt.bar(xpos, bic[i * len(n_components_range): (i + 1) * len(n_components_range)],
                            width=.2, color=color))
    # Do this to space evenly, even when the component range is not even
    plt.xticks(pos)
    plt.xlabel(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 + \
           .2 * np.floor(bic.argmin() / len(n_components_range))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    spl.set_xlabel('Number of components')
    spl.set_ylabel('BIC')
    spl.legend([b[0] for b in bars], cv_types)

    plt.xticks((n_components_range))
    plt.yticks(())
    plt.title('Selected GMM: {} cov, {} components'.format(best_cv, best_n_components))
    plt.subplots_adjust(hspace=.35, bottom=.02)
    plt.show()


if __name__ == '__main__':
    import datetime as dt

    skip = []
    start_time = dt.datetime(2018, 2, 7)
    rad = 'cvw'
    db_path = "./Data/cvw_GSoC_2018-02-07.db"

    for i in range(1):
        if i in skip:
            continue

        s = start_time + dt.timedelta(i)
        e = start_time + dt.timedelta(i + 1)
        data = read_db(db_path, rad, s, e)
        select_cov_and_clusters(data)