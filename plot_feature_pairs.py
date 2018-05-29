import cluster
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


""" slate for removal
def remove_outliers(arr):
    \"""
    Remove outliers for prettier plots
    :param arr: a 1D array or list
    :return: indices to remove the outliers
    \"""
    mean = np.mean(arr)
    std = np.std(arr)

    new_arr = [element for element in arr if (element > mean - 2 * std)]
    return new_arr
"""

def plot_feature_pairs(data_dict, show=True, save=False):
    data_flat, beam, gate, vel, wid, power, phi0, data_time = cluster.flatten_data(data_dict,  extras=True)

    feature_names = ["beam", "gate", "vel", "wid", "power", "phi0", "time"]

    """
    gate = gate ** 2      # RG = RG^2
    wid = np.sign(wid) * np.log(np.abs(wid))
    vel = np.sign(vel) * np.log(np.abs(vel))
    power = np.abs(power) ** 1.5
    """

    features = [beam, gate, vel, wid, power, phi0, data_time]

    gs_flg_gmm, clusters, median_vels_gmm = cluster.gmm(data_flat, vel, wid, num_clusters=6, cluster_identities=True)
    velocity_ordering = np.argsort(median_vels_gmm)
    clusters = [clusters[i] for i in velocity_ordering]
    median_vels_gmm = [median_vels_gmm[i] for i in velocity_ordering]
    cluster_labels = [("GS" if mvel < 15 else "IS") for mvel in median_vels_gmm]
    num_clusters = len(clusters)

    cluster_color = plt.cm.plasma(np.linspace(0, 1, num_clusters))

    plot_number = 0
    alpha = 0.1
    size = 1
    marker = 's'    #??

    """ Scatterplot """
    for ix in range(len(features)):
        for iy in range(ix+1, len(features)):
            #plot_clusters(features[ix], feature_names[ix], features[iy], feature_names[iy], clusters)

            plot_number += 1
            plt.figure(figsize=(4,4))
            x = features[ix]
            y = features[iy]

            plot_name = feature_names[ix] + " vs " + feature_names[iy]

            plt.title(plot_name)
            plt.xlabel(feature_names[ix])
            plt.ylabel(feature_names[iy])
            plt_x = x
            plt_y = y
            #TODO is this right? it was i, now it is ix
            plt.scatter(plt_x, plt_y, s=size, c=cluster_color[ix],
                        marker=marker, alpha=alpha, label=cluster_labels[ix])

            if show:
                plt.show()
            if save:
                plt.savefig(plot_number.__str__() + plot_name + start_time.__str__() + ".png")
                plt.close()


def plot_feature_pairs_by_cluster(data_dict, show=True, save=False):
    data_flat, beam, gate, vel, wid, power, phi0, data_time = cluster.flatten_data(data_dict,  extras=True)

    """
    gate = gate ** 2      # RG = RG^2
    wid = np.sign(wid) * np.log(np.abs(wid))
    vel = np.sign(vel) * np.log(np.abs(vel))
    power = np.abs(power) ** 1.5
    """
    features = [beam, gate, vel, wid, power, phi0, data_time]
    feature_names = ["beam", "gate", "vel", "wid", "power", "phi0", "time"]

    gs_flg_gmm, clusters, median_vels_gmm = cluster.gmm(data_flat, vel, wid, num_clusters=6, cluster_identities=True)
    velocity_ordering = np.argsort(median_vels_gmm)
    clusters = [clusters[i] for i in velocity_ordering]
    median_vels_gmm = [median_vels_gmm[i] for i in velocity_ordering]
    cluster_labels = [("GS" if mvel < 15 else "IS") for mvel in median_vels_gmm]
    num_clusters = len(clusters)

    cluster_color = plt.cm.plasma(np.linspace(0, 1, num_clusters))

    plot_number = 0
    alpha = 0.1
    size = 1
    marker = 's'    #??

    """ Scatterplot """
    for ix in range(len(features)):
        for iy in range(ix+1, len(features)):
            #plot_clusters(features[ix], feature_names[ix], features[iy], feature_names[iy], clusters)

            plot_number += 1
            x = features[ix]
            y = features[iy]

            plot_name = feature_names[ix] + " vs " + feature_names[iy]

            for i in range(num_clusters):
                plt.figure(figsize=(4, 4))
                plt.title(plot_name)
                plt.xlabel(feature_names[ix])
                plt.ylabel(feature_names[iy])
                plt_x = x[clusters[i]]
                plt_y = y[clusters[i]]
                plt.scatter(plt_x, plt_y, s=size, c=cluster_color[i],
                            marker=marker, alpha=alpha, label=cluster_labels[i])
                legend_handles = [mpatches.Patch(color=cluster_color[i], label=cluster_labels[i])]

                plt.legend(handles=legend_handles)
                if show:
                    plt.show()
                if save:
                    plt.savefig(plot_number.__str__() + plot_name + start_time.__str__() + " cluster " + i.__str__() + ".png")
                    plt.close()



if __name__ == '__main__':
    # Choose your date(s) and database
    skip = [20, 21, 22]
    start_time = dt.datetime(2017, 1, 17)
    database = "cvw"

    for i in range(1):
        if i in skip:
            continue
        s = start_time + dt.timedelta(i)
        e = start_time + dt.timedelta(i + 1)

        data = cluster.read_db(database, s, e)
        if not data:
            print('No data found')
            continue
        #plot_feature_pairs_by_cluster(data, show=False, save=True)
        plot_feature_pairs(data, show=False, save=True)