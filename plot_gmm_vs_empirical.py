from superdarn_cluster.cluster import *
from superdarn_cluster.utilities import plot_is_gs_colormesh
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from superdarn_cluster.dbtools import *
from sklearn.decomposition import PCA


def gmm_vs_empirical_colormesh(data_dict, start_time, end_time, radar='', clusters=6,
                               save=True, gmm_variation='PCA', pca_components=5):
    """
    Compare traditional, empirical, kmeans, and GMM

    Good info on different covariance types for GMM:
    https://stats.stackexchange.com/questions/326671/different-covariance-types-for-gaussian-mixture-models

    In the 'full' covariance matrix, the diagonal has the VARIANCE of the individual variables (often not 1), and the
    other values are the COVARIANCES. It should be symmetric around the diagonal.
    Each cluster has its own covariance matrix with 'full', so the shape of each one can be different.

    :param data_dict: dictionary from read_db
    :param start_time: datetime object
    :param end_time: datetime object
    :param save: if false, show plot, if save, create a .png file
    :param gmm_variation: 'PCA', 'Transformed', 'Variational Bayes'
    :return: creates a graph as PNG, title includes start_time
    """
    num_scatter = data_dict['num_scatter']
    data_flat, beam, gate, vel, wid, power, phi0, time_flat, filter = flatten_data(data_dict, extras=True, remove_close_range=True)
    pca_data_flat = np.column_stack((beam, gate, vel, wid, power, phi0, time_flat))
    pca_data_flat = pca_data_flat - np.mean(pca_data_flat, axis=0)
    trad_gs_flg = traditional(data_dict)[filter]

    # Mark indeterminate scatter in empirical (determined by negative values in the traditional GS flag)
    emp_gs_flg = empirical(data_dict)[filter]
    num_emp = len(emp_gs_flg)
    num_indeterminate = np.sum(emp_gs_flg == -1)
    percent_indet = float(num_indeterminate)/len(emp_gs_flg)*100
    remove_indeterminate = np.where(emp_gs_flg != -1)[0]

    trad_gs_flg = trad_gs_flg
    num_true_trad_gs = len(np.where((trad_gs_flg == 1) & (emp_gs_flg == 1))[0])
    num_true_trad_is = len(np.where((trad_gs_flg == 0) & (emp_gs_flg == 0))[0])
    accur_tra = float(num_true_trad_gs+num_true_trad_is)/num_emp*100.

    gmm_gs_flg = gmm(data_flat, vel, wid, num_clusters=clusters)
    num_true_gmm_gs = len(np.where((gmm_gs_flg == 1) & (emp_gs_flg == 1))[0]) #Assuming the GS is the cluster with minimum median velocity
    num_true_gmm_is = len(np.where((gmm_gs_flg == 0) & (emp_gs_flg == 0))[0])
    accur_gmm = float(num_true_gmm_gs+num_true_gmm_is)/num_emp*100.

    # Choice between several variations: transformed feature GMM, PCA GMM, variational bayes GMM
    if gmm_variation == 'Transformed':
        tran_gmm_data_flat, _ = flatten_data(data_dict, transform=True)
        tran_gmm_gs_flg = gmm(tran_gmm_data_flat, vel, wid, num_clusters=clusters)
    elif gmm_variation == 'PCA':
        pca = PCA(n_components=pca_components)
        pca.fit(data_flat)
        # Data_flat is already normalized to mean=0 standard dev=1, so we don't need to preprocess for PCA
        pca_data_flat = pca.transform(data_flat)
        tran_gmm_gs_flg = gmm(pca_data_flat, vel, wid, num_clusters=clusters)
    elif gmm_variation == 'Variational Bayes':
        #TODO
        return
    else:
        print('Bad gmm_variation flag ' + gmm_variation)
        return

    num_true_tran_gmm_gs = len(np.where((tran_gmm_gs_flg == 1) & (emp_gs_flg == 1))[0]) #Assuming the GS is the cluster with minimum median velocity
    num_true_tran_gmm_is = len(np.where((tran_gmm_gs_flg == 0) & (emp_gs_flg == 0))[0])
    accur_tran_gmm = float(num_true_tran_gmm_gs+num_true_tran_gmm_is)/num_emp*100.

    num_range_gates = data_dict['nrang'][0]
    time = data_dict['datetime']

    beams = np.unique(beam)
    time = np.array(time)
    for b in beams:
        scatter_flat = b == beam
        fig = plt.figure(figsize=(16, 8))
        ax1 = plt.subplot(411)
        plot_is_gs_colormesh(ax1, time, time_flat[scatter_flat], gate[scatter_flat], emp_gs_flg[scatter_flat],
                             num_range_gates, plot_indeterminate=True)
        ax1.set_title('Empirical Model Results [Burrell et al. 2015] Beam {} ({:3.2f}% flagged indeterminate)'.format(int(b), percent_indet))
        ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax1.set_xlim([start_time, end_time])

        ax2 = plt.subplot(412)
        plot_is_gs_colormesh(ax2, time, time_flat[scatter_flat], gate[scatter_flat], trad_gs_flg[scatter_flat], num_range_gates)
        ax2.set_title('Traditional Model Results [Blanchard et al. 2009] ({:3.2f}% agree with empirical)'.format(accur_tra))
        ax2.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax2.set_xlim([start_time, end_time])

        ax3 = plt.subplot(413)
        plot_is_gs_colormesh(ax3, time, time_flat[scatter_flat], gate[scatter_flat], gmm_gs_flg[scatter_flat], num_range_gates)
        ax3.set_title('Gaussian Mixture Model Results ({:3.2f}% agree with empirical)'.format(accur_gmm))
        ax3.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax3.set_xlim([start_time, end_time])

        ax4 = plt.subplot(414)
        plot_is_gs_colormesh(ax4, time, time_flat[scatter_flat], gate[scatter_flat], tran_gmm_gs_flg[scatter_flat], num_range_gates)
        ax4.set_title(gmm_variation + ' Gaussian Mixture Model Results ({:3.2f}% agree with empirical)'.format(accur_tran_gmm))
        ax4.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax4.set_xlim([start_time, end_time])

        fig.tight_layout()

        if save:
            plt.savefig(radar + " gmm vs empirical beam" + str(int(b)) + ".png")
            plt.close()
        else:
            plt.show()


if __name__ == '__main__':
    import datetime as dt

    skip = []
    start_time = dt.datetime(2018, 2, 7)
    rad = 'sas'
    db_path = "./Data/sas_GSoC_2018-02-07.db"

    for i in range(1):
        if i in skip:
            continue

        s = start_time + dt.timedelta(i)
        e = start_time + dt.timedelta(i + 1)
        data = read_db(db_path, rad, s, e)
        gmm_vs_empirical_colormesh(data, s, e, radar=rad, clusters=10, save=True, pca_components=5)
