from cluster import *
from utilities import plot_is_gs_colormesh
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter


def gmm_vs_empirical_colormesh(data_dict, start_time, end_time, clusters=6, show=True, save=False):
    """
    Compare traditional, empirical, kmeans, and GMM
    :param data_dict: dictionary from read_db
    :param start_time: datetime object
    :param end_time: datetime object
    :return: creates a graph as PNG, title includes start_time
    """
    trad_gs_flg = traditional(data_dict)
    emp_gs_flg = empirical(data_dict)
    data_flat, beam, gate, vel, wid, power, phi0, time_flat = flatten_data(data_dict, extras=True)

    remove_close_range = gate >= 1
    time_flat = time_flat[remove_close_range]
    gate = gate[remove_close_range]

    # Mark indeterminate scatter in empirical (determined by negative values in the traditional GS flag)
    emp_gs_flg = emp_gs_flg[remove_close_range]
    indeterminate = np.where(emp_gs_flg== -1)[0]
    emp_gs_flg[indeterminate] = -1
    num_emp = len(emp_gs_flg)
    percent_indet = float(len(indeterminate))/len(emp_gs_flg)*100

    trad_gs_flg = trad_gs_flg[remove_close_range]
    num_true_trad_gs = len(np.where((trad_gs_flg == 1) & (emp_gs_flg == 1))[0])
    num_true_trad_is = len(np.where((trad_gs_flg == 0) & (emp_gs_flg == 0))[0])
    accur_tra = float(num_true_trad_gs+num_true_trad_is)/num_emp*100.

    gmm_gs_flg = gmm(data_flat, vel, wid, num_clusters=clusters)
    gmm_gs_flg = gmm_gs_flg[remove_close_range]
    num_true_gmm_gs = len(np.where((gmm_gs_flg == 1) & (emp_gs_flg == 1))[0]) #Assuming the GS is the cluster with minimum median velocity
    num_true_gmm_is = len(np.where((gmm_gs_flg == 0) & (emp_gs_flg == 0))[0])
    accur_gmm = float(num_true_gmm_gs+num_true_gmm_is)/num_emp*100.

    tran_gmm_data_flat, _ = flatten_data(data_dict, transform=True)
    tran_gmm_gs_flg = gmm(tran_gmm_data_flat, vel, wid, num_clusters=clusters)
    tran_gmm_gs_flg = tran_gmm_gs_flg[remove_close_range]
    num_true_tran_gmm_gs = len(np.where((tran_gmm_gs_flg == 1) & (emp_gs_flg == 1))[0]) #Assuming the GS is the cluster with minimum median velocity
    num_true_tran_gmm_is = len(np.where((tran_gmm_gs_flg == 0) & (emp_gs_flg == 0))[0])
    accur_tran_gmm = float(num_true_tran_gmm_gs+num_true_tran_gmm_is)/num_emp*100.

    num_range_gates = data_dict['nrang'][0]
    time = data_dict['datetime']

    beams = np.unique(beam)
    time = np.array(time)
    for b in beams:

        fig = plt.figure(figsize=(20,8))
        ax1 = plt.subplot(411)
        plot_is_gs_colormesh(ax1, time, time_flat, gate, emp_gs_flg,
                             num_range_gates, plot_indeterminate=True)
        ax1.set_title('Empirical Model Results [Burrell et al. 2015] Beam {} ({:3.2f}% flagged indeterminate)'.format(int(b), percent_indet))
        ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax1.set_xlim([start_time, end_time])

        ax2 = plt.subplot(412)
        plot_is_gs_colormesh(ax2, time, time_flat, gate, trad_gs_flg, num_range_gates)
        ax2.set_title('Traditional Model Results [Blanchard et al. 2009] ({:3.2f}% agree with empirical)'.format(accur_tra))
        ax2.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax2.set_xlim([start_time, end_time])

        ax3 = plt.subplot(413)
        plot_is_gs_colormesh(ax3, time, time_flat, gate, gmm_gs_flg, num_range_gates)
        ax3.set_title('Gaussian Mixture Model Results ({:3.2f}% agree with empirical)'.format(accur_gmm))
        ax3.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax3.set_xlim([start_time, end_time])

        ax4 = plt.subplot(414)
        plot_is_gs_colormesh(ax4, time, time_flat, gate, tran_gmm_gs_flg, num_range_gates)
        ax4.set_title('Transformed Gaussian Mixture Model Results ({:3.2f}% agree with empirical)'.format(accur_tran_gmm))
        ax4.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax4.set_xlim([start_time, end_time])

        filename = int(b).__str__() + " gmm vs. trad vs. emp colormesh " + start_time.__str__() + ".png"
        fig.tight_layout()
        if show or b == 1:
            plt.show()
        if save:
            plt.savefig(filename)
            plt.close()


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
        gmm_vs_empirical_colormesh(data, s, e, clusters=2, show=False, save=True)