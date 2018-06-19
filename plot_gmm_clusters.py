from superdarn_cluster.cluster import *
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import datetime as dt
from superdarn_cluster.dbtools import *
from sklearn.decomposition import PCA

def test(asdf, bsdf):
    print(asdf)
    print(bsdf)
    return

def plot_gmm_clusters(data_flat, data_flat_unscaled, time, gate, vel, feature_names, range_max, start_time, end_time,
                      num_clusters=6, radar='', save=True, use_pickle=False, save_pickle=False, picklefile="",):
    """
    Plots GMM against empirical, but with color-coding for the various clusters

    :param data_dict:
    :param start_time:
    :param end_time:
    :return:
    """
    #data_flat, beam, gate, vel, wid, power, phi0, data_time, filter = flatten_data(data_dict, extras=True, remove_close_range=True)
    #data_flat_unscaled = np.column_stack((beam, gate, vel, wid, power, phi0, data_time))
    #range_max = data_dict['nrang'][0]

    """ Do GMM """

    if use_pickle:
        pfile = open("./GMMPickles/" + picklefile, 'rb')
        cluster_membership = pickle.load(pfile)
    else:
        estimator = GaussianMixture(n_components=num_clusters,
                                    covariance_type='full', max_iter=500,
                                    random_state=0, n_init=5, init_params='kmeans')
        estimator.fit(data_flat)
        cluster_membership = estimator.predict(data_flat)

        if save_pickle:
            picklefile = "./GMMPickles/" + picklefile
            pfile = open(picklefile, 'wb')
            pickle.dump(cluster_membership, pfile)

    gs_class_gmm = []
    is_class_gmm = []

    median_vels_gmm = np.zeros(num_clusters)
    max_vels_gmm = np.zeros(num_clusters)
    min_vels_gmm = np.zeros(num_clusters)

    for i in range(num_clusters):
        median_vels_gmm[i] = np.median(np.abs(vel[cluster_membership == i]))
        max_vels_gmm[i] = np.max(np.abs(vel[cluster_membership == i]))
        min_vels_gmm[i] = np.min(np.abs(vel[cluster_membership == i]))

        if median_vels_gmm[i] > 15:
            is_class_gmm.append(i)
        else:
            gs_class_gmm.append(i)

    cluster_membership = cluster_membership

    gs_flg_gmm = []
    for i in cluster_membership:
        if i in gs_class_gmm:
            gs_flg_gmm.append(1)
        elif i in is_class_gmm:
            gs_flg_gmm.append(0)

    gs_flg_gmm = np.array(gs_flg_gmm)
    clusters = [np.where(cluster_membership == i)[0] for i in range(num_clusters)]

    cm = plt.cm.get_cmap('coolwarm')
    alpha = 0.15
    size = 1
    marker = 's'

    """ Plot Individual Clusters """
    # Do color coding by cluster
    # TODO https://stackoverflow.com/questions/19064772/visualization-of-scatter-plots-with-overlapping-points-in-matplotlib
    #cluster_col = plt.cm.viridis(np.linspace(0, 1, num_clusters))
    cluster_col = plt.cm.plasma(np.linspace(0, 1, num_clusters))
    alpha = 1
    # Plot individually
    for i in range(num_clusters):
        plt.figure(figsize=(16, 8))
        ax0 = plt.subplot(211)
        plt.scatter(time[clusters[i]], gate[clusters[i]],
                    s=size, c=cluster_col[i], marker=marker, alpha=alpha, label=median_vels_gmm[i])
        ax0.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax0.set_xlim([start_time, end_time])
        ax0.set_ylim([0, range_max])
        ax0.set_ylabel('Range gate')
        ax0.set_title('Gaussian Mixture Model Cluster' + (i+1).__str__())

        # Make statistics table
        var = np.var(data_flat_unscaled[clusters[i], :], axis=0)
        median = np.median(data_flat_unscaled[clusters[i], :], axis=0)
        mean = np.mean(data_flat_unscaled[clusters[i], :], axis=0)
        max = np.max(data_flat_unscaled[clusters[i], :], axis=0)
        min = np.min(data_flat_unscaled[clusters[i], :], axis=0)
        stats = np.array([var, median, mean, max, min])

        ax1 = plt.subplot(212)
        ax1.axis('off')
        stats_table = ax1.table(cellText=stats, rowLabels=['var', 'med', 'mean', 'max', 'min'], colLabels=feature_names, loc='center')

        # Print some statistics
        """
        var = np.var(data_flat_unscaled[clusters[i], :], axis=0)
        median = np.median(data_flat_unscaled[clusters[i], :], axis=0)
        mean = np.mean(data_flat_unscaled[clusters[i], :], axis=0)
        max = np.max(data_flat_unscaled[clusters[i], :], axis=0)
        min = np.min(data_flat_unscaled[clusters[i], :], axis=0)
        stats = np.array([var, median, mean, max, min])
        stats_df = pd.DataFrame(data=stats, columns=feature_names, index=['var', 'med', 'mean', 'max', 'min'])
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            plottext = stats_df.to_string()
            print(plottext)


        stats = feature_names
        #stats = "Velocity:\n median {:3.1f}\n max {:3.1f}\n min {:3.1f}\n".format(
        #    median_vels_gmm[i], max_vels_gmm[i], min_vels_gmm[i])
        #stats += "Spectral width:\n median {:3.1f}\n max {:3.1f}\n min {:3.1f}\n".format(
        #    median_wids_gmm[i], max_wids_gmm[i], min_wids_gmm[i])
        # First arg is left-right 0-1, Second arg is down-up 0-1
        plt.subplots_adjust(left=0.1, bottom=0.5, right=0.9, top=0.9, wspace=0, hspace=0)
        plt.gcf().text(0.3, 0.2, plottext, fontsize=10)
        #plt.subplots_adjust(right=0.85, left=0.1)
        """

        if save:
            plt.savefig(radar + " individual cluster " + (i+1).__str__() + ".png")
            plt.close()
        else:
            plt.show()
            pass


    """ Plot All Clusters """
    alpha = 0.15
    plot_number = num_clusters + 1
    plt.figure(figsize=(15, 6))
    ax0 = plt.subplot(111)
    ax0.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax0.set_xlim([start_time, end_time])
    ax0.set_ylabel('Range gate')
    ax0.set_ylim(0, range_max)
    ax0.set_title('Gaussian Mixture Model All Clusters')
    cluster_labels = [("GS" if mvel < 15 else "IS") for mvel in median_vels_gmm]
    legend_handles = []
    for i in range(num_clusters):
        plt.scatter(time[clusters[i]], gate[clusters[i]], s=size, c=cluster_col[i],
                    marker=marker, alpha=alpha, label=cluster_labels[i])
        legend_handles.append(mpatches.Patch(color=cluster_col[i], label=cluster_labels[i]))

    plt.legend(handles=legend_handles)
    if save:
        plt.savefig("individual clusters all together scatterplot.png")
        plt.close()
    else:
        plt.show()

    """ Plot RTI 
    # plot_rti_mesh(data_dict, clusters, start_time, end_time, cluster_order=list(np.argsort(median_vels_gmm)))
    # def plot_rti_mesh(data_dict, clusters, start_time, end_time, cluster_order=[]):

    # number of range gate, usually = 75 or 110
    range_max = data_dict['nrang'][0]

    num_times = len(data_dict['datetime'])
    times = date2num(data_dict['datetime'])
    color_mesh = np.zeros((num_times, range_max)) * np.nan
    num_clusters = len(clusters)
    plot_param = 'velocity'

    times_flat = []
    num_scatter = data_dict['num_scatter']
    for i in range(len(num_scatter)):
        times_flat.extend(date2num([data_dict['datetime'][i]] * num_scatter[i]))

    # Create a (num times) x (num range gates) map of cluster values.
    # The colormap will then plot those values as cluster values.
    # cluster_color = np.linspace(-200, 200, num_clusters)
    cluster_order = np.argsort(median_vels_gmm)
    color = 0
    print('RTI plot cluster memberships')
    for k in cluster_order:
        # Cluster membership indices correspond to the flattened data, which may contain repeat time values
        for i in clusters[k]:
            time = times_flat[i]
            ii = np.where(time == times)[0][0]
            gates = data_dict['gate'][ii]

            for g in gates:
                color_mesh[ii, g] = color
        color += 1
        IS = median_vels_gmm[k] > 10
        if IS:
            print(color, "IS")
        else:
            print(color, "GS")

    # Create a matrix of the right size
    range_gate = np.linspace(1, range_max, range_max)
    mesh_x, mesh_y = np.meshgrid(times, range_gate)
    invalid_data = np.ma.masked_where(np.isnan(color_mesh.T), color_mesh.T)
    # Zm = np.ma.masked_where(np.isnan(data[:tcnt][:].T), data[:tcnt][:].T)
    # Set colormap so that masked data (bad) is transparent.

    cmap, norm, bounds = util.genCmap(plot_param, [0, len(clusters)],
                                                 colors='lasse',
                                                 lowGray=False)
    cmap.set_bad('w', alpha=0.0)

    pos = [.1, .1, .76, .72]
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_axes(pos)
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax.set_xlabel('UT')
    ax.set_xlim([start_time, end_time])
    ax.set_ylabel('Range gate')
    # ax.set_title(title_str+' '+start_time.strftime("%d %b %Y") + ' ' + rad.upper())
    colormesh = ax.pcolormesh(mesh_x, mesh_y, invalid_data, lw=0.01, edgecolors='None', cmap=cmap, norm=norm)

    # Draw the colorbar.
    cb = util.drawCB(fig, colormesh, cmap, norm, map_plot=0,
                      pos=[pos[0] + pos[2] + .02, pos[1], 0.02, pos[3]])

    if save:
        plt.savefig("individual clusters all together colormesh.png")
        plt.close()
    else:
        plt.show()
    """


if __name__ == '__main__':
    skip = []
    start_time = dt.datetime(2018, 2, 7)
    rad = 'sas'
    db_path = "./Data/sas_GSoC_2018-02-07.db"
    transform = False

    for i in range(1):
        if i in skip:
            continue

        s = start_time + dt.timedelta(i)
        e = start_time + dt.timedelta(i + 1)

        data = read_db(db_path, rad, s, e)
        feature_names = ['beam','gate','vel','wid','power','phi0','time']
        data_flat, beam, gate, vel, wid, power, phi0, time, filter = flatten_data(data, extras=True, remove_close_range=True)
        data_flat_unscaled = np.column_stack((beam, gate, np.abs(vel), np.abs(wid), power, phi0, time))
        range_max = data['nrang'][0]

        plot_gmm_clusters(data_flat, data_flat_unscaled, time, gate, vel, feature_names,
                          range_max, s, e, num_clusters=10, radar=rad)
