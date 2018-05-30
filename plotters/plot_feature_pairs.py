import cluster
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from dbtools import *

# TODO add some statistical information to plot by cluster
# TODO you could do a colormesh for the combined graph, but the scales are so varying it's almost pointless


def plot_feature_pairs_by_cluster(data_dict, num_clusters=6, save=True):
    """

    :param data_dict:
    :param save: If false, plots will pop up - if true, they will be saved.
                 Recommend saving because this puts out 21*num_clusters plots.
    :return:
    """
    data_flat, beam, gate, vel, wid, power, phi0, data_time = flatten_data(data_dict,  extras=True)

    """
    gate = gate ** 2      # RG = RG^2
    wid = np.sign(wid) * np.log(np.abs(wid))
    vel = np.sign(vel) * np.log(np.abs(vel))
    power = np.abs(power) ** 1.5
    """
    features = [beam, gate, vel, wid, power, phi0, data_time]
    feature_names = ["beam", "gate", "vel", "wid", "power", "phi0", "time"]

    gs_flg_gmm, clusters, median_vels_gmm = cluster.gmm(data_flat, vel, wid, num_clusters=num_clusters, cluster_identities=True)
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

                if save:
                    plt.savefig(plot_number.__str__() + plot_name + start_time.__str__() + " cluster " + i.__str__() + ".png")
                    plt.close()
                else:
                    plt.show()


if __name__ == '__main__':
    # Choose your date(s) and database
    skip = []
    start_time = dt.datetime(2018, 2, 7)
    db_path = "../Data/cvw_GSoC_2018-02-07.db"
    rad = "cvw"

    for i in range(1):
        if i in skip:
            continue
        s = start_time + dt.timedelta(i)
        e = start_time + dt.timedelta(i + 1)

        data = read_db(db_path, rad, s, e)
        plot_feature_pairs_by_cluster(data)
