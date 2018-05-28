from cluster import *


def compare_pca(data_dict, date_str):
    gate = np.hstack(data_dict['gate'])
    vel = np.hstack(data_dict['velocity'])
    wid = np.hstack(data_dict['width'])
    data_flat, time = flatten_data(data_dict)

    gs_flg_kmeans = kmeans(data_flat, vel, wid)
    gs_flg_kmeans_pca = kmeans(data_flat, vel, wid, pca=True)

    plot_scatter(time, gate, gs_flg_kmeans, "KMeans results " + date_str)
    plot_scatter(time, gate, gs_flg_kmeans_pca, "KMeans + PCA results " + date_str)

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

    print('The GS/IS identification accurary of kmeans is {:3.2f}%'.format(accur_kmeans))
    print('The GS/IS identification accurary of PCA kmeans is {:3.2f}%'.format(accur_kmeans_pca))

    """
    gs_flg_gmm = gmm(data_flat, vel, wid)
    gs_flg_gmm_pca = gmm(data_flat, vel, wid, pca=True)

    num_true_gmm_gs = len(np.where((gs_flg_gmm == 1) & (emp_gs_flg == 1))[0])
    num_true_gmm_is = len(np.where((gs_flg_gmm == 0) & (emp_gs_flg == 0))[0])
    accur_gmm = float(num_true_gmm_gs+num_true_gmm_is)/num_emp*100.
    print 'The GS/IS identification accurary of GMM is {:3.2f}%'.format(accur_gmm)

    num_true_gmm_pca_gs = len(np.where((gs_flg_gmm_pca == 1) & (emp_gs_flg == 1))[0])
    num_true_gmm_pca_is = len(np.where((gs_flg_gmm_pca == 0) & (emp_gs_flg == 0))[0])
    accur_gmm_pca = float(num_true_gmm_pca_gs+num_true_gmm_pca_is)/num_emp*100.
    print 'The GS/IS identification accurary of PCA GMM is {:3.2f}%'.format(accur_gmm_pca)
    """