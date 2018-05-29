from cluster import flatten_data, kmeans, empirical, gmm, read_db
import numpy as np
from utilities import plot_is_gs_scatterplot

def compare_pca(data_dict, num_clusters=6):
    """
    Plot K-means and GMM with and without PCA
    :param data_dict:
    :param num_clusters:
    :return:
    """
    #TODO - this script is still in bad shape, needs updating before use
    #TODO - add plots for both GMM and kmeans, use colormesh instead of scatter

    gate = np.hstack(data_dict['gate'])
    vel = np.hstack(data_dict['velocity'])
    wid = np.hstack(data_dict['width'])
    data_flat, time = flatten_data(data_dict)

    gs_flg_kmeans = kmeans(data_flat, vel, wid, num_clusters=num_clusters)
    gs_flg_kmeans_pca = kmeans(data_flat, vel, wid, num_clusters=num_clusters, pca=True)

    plot_is_gs_scatterplot(time, gate, gs_flg_kmeans, "KMeans results ")
    plot_is_gs_scatterplot(time, gate, gs_flg_kmeans_pca, "KMeans + PCA results")

    # Compare accuracy against empirical method
    emp_gs_flg, emp_time, emp_gate = empirical(data_dict)
    num_emp = len(emp_gs_flg)

    num_true_kmeans_gs = len(np.where((gs_flg_kmeans == 1) & (emp_gs_flg == 1))[
                                 0])  # Assuming the GS is the cluster with minimum median velocity
    num_true_kmeans_is = len(np.where((gs_flg_kmeans == 0) & (emp_gs_flg == 0))[0])
    accur_kmeans = float(num_true_kmeans_gs + num_true_kmeans_is) / num_emp * 100.

    num_true_kmeans_pca_gs = len(np.where((gs_flg_kmeans_pca == 1) & (emp_gs_flg == 1))[
                                     0])  # Assuming the GS is the cluster with minimum median velocity
    num_true_kmeans_pca_is = len(np.where((gs_flg_kmeans_pca == 0) & (emp_gs_flg == 0))[0])
    accur_kmeans_pca = float(num_true_kmeans_pca_gs + num_true_kmeans_pca_is) / num_emp * 100.

    print('Kmeans ({:3.2f}% agree with empirical)'.format(accur_kmeans))
    print('PCA Kmeans ({:3.2f}% agree with empirical)'.format(accur_kmeans_pca))

    gs_flg_gmm = gmm(data_flat, vel, wid, num_clusters=num_clusters)
    gs_flg_gmm_pca = gmm(data_flat, vel, wid, num_clusters=num_clusters, pca=True)

    num_true_gmm_gs = len(np.where((gs_flg_gmm == 1) & (emp_gs_flg == 1))[0])
    num_true_gmm_is = len(np.where((gs_flg_gmm == 0) & (emp_gs_flg == 0))[0])
    accur_gmm = float(num_true_gmm_gs+num_true_gmm_is)/num_emp*100.
    print('GMM ({:3.2f}% agree with empirical)'.format(accur_gmm))

    num_true_gmm_pca_gs = len(np.where((gs_flg_gmm_pca == 1) & (emp_gs_flg == 1))[0])
    num_true_gmm_pca_is = len(np.where((gs_flg_gmm_pca == 0) & (emp_gs_flg == 0))[0])
    accur_gmm_pca = float(num_true_gmm_pca_gs+num_true_gmm_pca_is)/num_emp*100.
    print('PCA GMM ({:3.2f}% agree with empirical)'.format(accur_gmm_pca))


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
        compare_pca(data, num_clusters=2)
