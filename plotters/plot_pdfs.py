from cluster import gmm
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from matplotlib.dates import DateFormatter
from dbtools import *

def plot_pdfs_overlay(data_dict, start_time, num_clusters=6, save=True):
    """
    :param data_dict:
    :param start_time:
    :param end_time:
    :param num_clusters:
    :param show:
    :param save: if false, show plot, if save, create a .png file
    :return:
    """
    data_flat, beam, gate, vel, wid, power, phi0, data_time = flatten_data(data_dict, extras=True)
    vel = np.abs(vel)
    remove_close_range = gate >= 10
    data_flat_columns = ['beam', 'gate', 'vel', 'wid', 'power', 'phi0', 'time']

    """
    g_gate = gate ** 2  # RG = RG^2
    g_wid = np.sign(wid) * np.log(np.abs(wid))
    g_vel = np.sign(vel) * np.log(np.abs(vel))
    g_power = np.abs(power) ** 1.5
    data_flat_unscaled = np.column_stack((beam, g_gate, g_vel, g_wid, g_power, phi0, data_time))
    """

    data_flat_unscaled = np.column_stack((beam, gate, vel, wid, power, phi0, data_time))
    data_flat_unscaled = data_flat_unscaled[remove_close_range, :]

    gs_flg_gmm = gmm(data_flat, vel, wid, num_clusters=num_clusters)

    gs_data = data_flat_unscaled[gs_flg_gmm[remove_close_range] == 1]
    #gs_data[:, 2] = np.abs(gs_data[:, 2])
    is_data = data_flat_unscaled[gs_flg_gmm[remove_close_range] == 0]
    #is_data[:, 2] = np.abs(is_data[:, 2])
    plot_number = 1

    # Plot a separate histogram for each feature
    for i in [2]: #range(data_flat.shape[1]):
        plt.figure(figsize=(15, 6))
        ax0 = plt.subplot(111)
        ax0.set_xlabel(data_flat_columns[i])
        ax0.set_ylabel('pdf')
        ax0.set_title('probability density for ' + data_flat_columns[i])

        gs_num_bins = len(np.unique(gs_data[:, i]))
        if gs_num_bins:
            if gs_num_bins > 1000:
                gs_num_bins = 1000
            gs_y, gs_binedges = np.histogram(gs_data[:, i], bins=gs_num_bins)
            gs_bincenters = 0.5 * (gs_binedges[1:] + gs_binedges[:-1])
            gs_pdf = gs_y / float(len(gs_data))
            plt.plot(gs_bincenters, gs_pdf, 'b', label='GS')

        is_num_bins = len(np.unique(is_data[:, i]))
        if is_num_bins:
            if is_num_bins > 1000:
                is_num_bins = 1000
            is_y, is_binedges = np.histogram(is_data[:, i], bins=is_num_bins)
            is_bincenters = 0.5 * (is_binedges[1:] + is_binedges[:-1])
            is_pdf = is_y / float(len(is_data))
            plt.plot(is_bincenters, is_pdf, 'r', label='IS')

        # gs_threshhold = gs_pdf > 0.0008
        # is_threshhold = is_pdf > 0.0008
        # plt.xlim(xmin=0)
        # plt.xlim(xmax=650)
        # plt.ylim(ymin=0)
        plot_number += 1
        plt.legend()

        # legend_handles.append(mpatches.Patch(color=cluster_col[i], label=cluster_labels[i]))
        # plt.legend(handles=legend_handles)
        if save:
            plt.savefig(str(plot_number) + '_' + data_flat_columns[i] + "_GMM_histogram" + start_time.__str__() + ".png")
            plt.close()
        else:
            plt.show()


def plot_pdfs(data_dict, start_time, num_clusters=6, save=False):

    data_flat, beam, gate, vel, wid, power, phi0, data_time = flatten_data(data_dict, extras=True)
    remove_close_range = gate >= 10
    data_flat_columns = ['beam', 'gate', 'vel', 'wid', 'power', 'phi0', 'time']

    tran_data_flat, _ = flatten_data(data_dict, transform=True)
    g_gate = gate ** 2  # RG = RG^2
    g_wid = np.sign(wid) * np.log(np.abs(wid))
    g_vel = np.sign(vel) * np.log(np.abs(vel))
    g_power = np.log(power) #np.abs(power) ** 1.5
    tran_data_flat_unscaled = np.column_stack((beam, g_gate, g_vel, g_wid, g_power, phi0, data_time))
    tran_data_flat_unscaled = tran_data_flat_unscaled[remove_close_range]
    gs_flg_tran_gmm = gmm(data_flat, vel, wid, num_clusters=num_clusters)
    tran_gs_data = tran_data_flat_unscaled[gs_flg_tran_gmm[remove_close_range] == 1]
    tran_is_data = tran_data_flat_unscaled[gs_flg_tran_gmm[remove_close_range] == 0]

    transformed = [1, 2, 3, 4]

    data_flat_unscaled = np.column_stack((beam, gate, vel, wid, power, phi0, data_time))
    data_flat_unscaled = data_flat_unscaled[remove_close_range, :]
    gs_flg_gmm = gmm(data_flat, vel, wid, num_clusters=num_clusters)
    gs_data = data_flat_unscaled[gs_flg_gmm[remove_close_range] == 1]
    is_data = data_flat_unscaled[gs_flg_gmm[remove_close_range] == 0]

    highres_bin_limit = 200
    lowres_bin_limit = 200

    # Plot a separate histogram for each feature
    plot_number = 1
    for i in [1]: #range(data_flat.shape[1]):
        plt.figure(figsize=(12,10))

        ax0 = plt.subplot(321)
        #ax.set_xlabel(data_flat_columns[i])
        ax0.set_ylabel('pdf')
        ax0.set_title('Probability density for ' + data_flat_columns[i])
        num_bins = len(np.unique(data_flat_unscaled[:, i]))
        if num_bins > highres_bin_limit:
            num_bins = highres_bin_limit
        ax0.hist(data_flat_unscaled[:, i], bins=num_bins, density=True, color='orange')

        ax1 = plt.subplot(323)
        #ax0.set_xlabel(data_flat_columns[i])
        ax1.set_ylabel('pdf')
        ax1.set_title('GMM output: GS probability density for ' + data_flat_columns[i])

        gs_num_bins = len(np.unique(gs_data[:, i]))
        if gs_num_bins:
            if gs_num_bins > lowres_bin_limit:
                gs_num_bins = lowres_bin_limit
            ax1.hist(gs_data[:, i], bins=gs_num_bins, density=True, color='b')
        """
            gs_y, gs_binedges = np.histogram(gs_data[:, i], bins=gs_num_bins)
            gs_bincenters = 0.5 * (gs_binedges[1:] + gs_binedges[:-1])
            gs_pdf = gs_y / float(len(gs_data))
            plt.plot(gs_bincenters, gs_pdf, 'b', label='GS')
        """

        ax2 = plt.subplot(325)
        #ax1.set_xlabel(data_flat_columns[i])
        ax2.set_ylabel('pdf')
        ax2.set_title('GMM output: IS probability density for ' + data_flat_columns[i])
        is_num_bins = len(np.unique(is_data[:, i]))
        if is_num_bins:
            if is_num_bins > highres_bin_limit:
                is_num_bins = highres_bin_limit
            ax2.hist(is_data[:, i], bins=is_num_bins, density=True, color='r')
        """
            is_y, is_binedges = np.histogram(is_data[:, i], bins=is_num_bins)
            is_bincenters = 0.5 * (is_binedges[1:] + is_binedges[:-1])
            is_pdf = is_y / float(len(is_data))
            plt.plot(is_bincenters, is_pdf, 'r', label='IS')
        """

        # gs_threshhold = gs_pdf > 0.0008
        # is_threshhold = is_pdf > 0.0008
        # plt.xlim(xmin=0)
        # plt.xlim(xmax=650)
        # plt.ylim(ymin=0)
        plot_number += 1
        plt.legend()


        if i in transformed:
            ax3 = plt.subplot(322)
            # ax.set_xlabel(data_flat_columns[i])
            ax3.set_ylabel('pdf')
            ax3.set_title('Probability density for ' + data_flat_columns[i])
            num_bins = len(np.unique(tran_data_flat_unscaled[:, i]))
            if num_bins > lowres_bin_limit:
                num_bins = lowres_bin_limit
            ax3.hist(tran_data_flat_unscaled[:, i], bins=num_bins, density=True, color='orange')

            # Transformed data
            ax4 = plt.subplot(324)
            ax4.set_xlabel(data_flat_columns[i])
            ax4.set_ylabel('pdf')
            ax4.set_title('GMM output: GS probability density for transformed ' + data_flat_columns[i])

            tran_gs_num_bins = len(np.unique(tran_gs_data[:, i]))
            if tran_gs_num_bins:
                if tran_gs_num_bins > lowres_bin_limit:
                    tran_gs_num_bins = lowres_bin_limit
                ax4.hist(tran_gs_data[:, i], bins=tran_gs_num_bins, density=True, color='b')

            ax5 = plt.subplot(326)
            ax5.set_xlabel(data_flat_columns[i])
            ax5.set_ylabel('pdf')
            ax5.set_title('GMM output: IS probability density for transformed ' + data_flat_columns[i])

            tran_is_num_bins = len(np.unique(tran_is_data[:, i]))
            if tran_is_num_bins:
                if tran_is_num_bins > lowres_bin_limit:
                    tran_is_num_bins = lowres_bin_limit
                ax5.hist(tran_is_data[:, i], bins=tran_is_num_bins, density=True, color='red')

        if data_flat_columns[i] == 'time':
            ax0.xaxis.set_major_formatter(DateFormatter('%H:%M'))
            ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))
            ax2.xaxis.set_major_formatter(DateFormatter('%H:%M'))
            ax3.xaxis.set_major_formatter(DateFormatter('%H:%M'))
            ax4.xaxis.set_major_formatter(DateFormatter('%H:%M'))
            ax5.xaxis.set_major_formatter(DateFormatter('%H:%M'))


        plt.tight_layout()

        if save:
            plt.savefig(str(plot_number) + '_' + data_flat_columns[i] + "_GMM_histogram" + start_time.__str__() + ".png")
            plt.close()
        else:
            plt.show()


if __name__ == '__main__':
    skip = []
    start_time = dt.datetime(2018, 2, 7)
    rad = 'cvw'
    db_path = "../Data/cvw_GSoC_2018-02-07.db"
    transform = False

    for i in range(1):
        if i in skip:
            continue

        s = start_time + dt.timedelta(i)
        e = start_time + dt.timedelta(i + 1)

        data = read_db(db_path, rad, s, e)
        if not data:
            print('No data found')
            continue
        plot_pdfs(data, s, num_clusters=2, save=False)