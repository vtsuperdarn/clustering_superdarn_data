# Author: Ron Weiss <ronweiss@gmail.com>, Gael Varoquaux
# Modified by Thierry Guillemot <thierry.guillemot.work@gmail.com>
# License: BSD 3 clause

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from superdarn_cluster.cluster import gmm
from superdarn_cluster.dbtools import flatten_data
import datetime as dt
from superdarn_cluster.dbtools import read_db
from math import ceil
from sklearn.decomposition import PCA

# TODO add some statistical information to the graphs - cluster mean/var

def make_ellipses(model, ax, colors, n_cluster, f1, f2):
    """
    Plot an ellipse representing one cluster in GMM.

    Note: This ellipse will be centered at the mean, and its size represents the VARIANCE,
    not the STANDARD DEVIATION. Variance is taken from the diagonal of the covariance matrix.

    :param model: a GMM model trained on some data
    :param ax: the subplot axis to draw on
    :param colors: a list of colors (length num_clusters)
    :param n_cluster: integer cluster index, in range [0, num_clousters]
    :param f1: feature 1 index, in range [0, num_features]
    :param f2: feature 2 index, in range [0, num_features]
    """
    if model.covariance_type == 'full':
        covariances = model.covariances_[n_cluster][[f1, f2], :][:, [f1, f2]]
    elif model.covariance_type == 'tied':
        covariances = model.covariances_[[f1, f2], :][:, [f1, f2]]
    # TODO this may or may not work
    elif model.covariance_type == 'diag':
        covariances = np.diag(model.covariances_[n_cluster][[f1, f2]])
    # TODO will this work...? do we need it?
    elif model.covariance_type == 'spherical':
        covariances = []  # np.eye(model.means_.shape[1]) * model.covariances_[n_cluster]

    v, w = np.linalg.eigh(covariances)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    means = model.means_[n_cluster, [f1, f2]]

    print('Covariance')
    print(covariances)
    print('Mean')
    print(means)

    ell = mpl.patches.Ellipse(means, v[0], v[1],
                              180 + angle, color='black')
    ell.set_clip_box(ax.bbox)
    ell.set_alpha(0.5)
    ax.add_artist(ell)


def plot_feature_pairs_by_cluster(data_dict, num_clusters=6, save=True, gmm_variation=None):

    data_flat, beam, gate, vel, wid, power, phi0, data_time = flatten_data(data_dict,  extras=True)

    """
    gate = gate ** 2      # RG = RG^2
    wid = np.sign(wid) * np.log(np.abs(wid))
    vel = np.sign(vel) * np.log(np.abs(vel))
    power = np.abs(power) ** 1.5
    """

    if gmm_variation == 'PCA':
        # Do PCA
        pca = PCA(n_components=7)
        pca.fit(data_flat)
        data_flat = pca.transform(data_flat)

        num_features = data_flat.shape[1]
        feature_names = ["PC" + str(i) for i in range(num_features)]

    else:
        num_features = data_flat.shape[1]
        feature_names = ["beam", "gate", "vel", "wid", "power", "phi0", "time"]

    gs_flg_gmm, clusters, median_vels_gmm, estimator = gmm(data_flat, vel, wid, num_clusters=num_clusters, cluster_identities=True)
    cluster_ids = estimator.predict(data_flat)
    velocity_ordering = np.argsort(median_vels_gmm)
    clusters = [clusters[i] for i in velocity_ordering]
    median_vels_gmm = [median_vels_gmm[i] for i in velocity_ordering]
    cluster_labels = [("GS" if mvel < 15 else "IS") for mvel in median_vels_gmm]
    num_clusters = len(clusters)
    colors = plt.cm.plasma(np.linspace(0, 1, num_clusters))

    for f1 in range(num_features):
        for f2 in range(f1+1, num_features):
            plt.figure(figsize=(20, 12))
            plot_name = feature_names[f1] + " vs " + feature_names[f2]

            for c in range(num_clusters):
                print('===========================')
                print('cluster', c, 'features', f1, f2)
                print('===========================')
                print(feature_names[f1], 'mean', np.mean(data_flat[cluster_ids == c, f1]), 'var', np.var(data_flat[cluster_ids == c, f1]))
                print(feature_names[f2], 'mean', np.mean(data_flat[cluster_ids == c, f2]), 'var', np.var(data_flat[cluster_ids == c, f2]))
                print()

                ax = plt.subplot(2, ceil(num_clusters / 2.0), c + 1)
                ax.scatter(data_flat[cluster_ids == c, f1], data_flat[cluster_ids == c, f2],
                           alpha=0.1, marker='x', color=colors[c], label=cluster_labels[c])

                make_ellipses(estimator, ax, colors, c, f1, f2)

                plt.legend()

                plt.xlabel(feature_names[f1])
                plt.ylabel((feature_names[f2]))
                plt.title(plot_name)

            if save:
                plt.savefig(plot_name + '.png')
                plt.close()
            else:
                plt.show()


if __name__ == '__main__':
    # Choose your date(s) and database
    skip = []
    start_time = dt.datetime(2018, 2, 7)
    db_path = "./Data/cvw_GSoC_2018-02-07.db"
    rad = "cvw"

    for i in range(1):
        if i in skip:
            continue
        s = start_time + dt.timedelta(i)
        e = start_time + dt.timedelta(i + 1)

        data = read_db(db_path, rad, s, e)
        plot_feature_pairs_by_cluster(data, num_clusters=10)