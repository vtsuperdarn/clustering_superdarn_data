from cluster import *
import matplotlib.patches as mpatches
import utilities
import datetime as dt

def plot_gmm_clusters(data_dict, start_time, end_time, num_clusters=5, show=True, save=False, use_pickle=False, save_pickle=False):
    """
    Plots GMM against empirical, but with color-coding for the various clusters

    :param data_dict:
    :param start_time:
    :param end_time:
    :return:
    """
    data_flat, beam, gate, vel, wid, power, phi0, data_time = flatten_data(data_dict, extras=True)

    """ Do Empirical Method"""
    emp_gs_flg, emp_time, emp_gate = empirical(data_dict)
    remove_close_range = gate >= 10
    num_emp = len(emp_gs_flg[remove_close_range])

    """ Do GMM """
    if use_pickle:
        picklefile = open("./GMMPickles/gmm" + num_clusters.__str__() + '_' + start_time.__str__() + ".pickle", 'r')
        cluster_membership = pickle.load(picklefile)
    else:
        estimator = GaussianMixture(n_components=num_clusters,
                                covariance_type='full', max_iter=500,
                                random_state=0, n_init=5, init_params='kmeans')
        estimator.fit(data_flat)
        cluster_membership = estimator.predict(data_flat)

    if save_pickle:
        picklefile = open("./GMMPickles/gmm" + num_clusters.__str__() + '_' + start_time.__str__() + ".pickle", 'w')
        pickle.dump(cluster_membership, picklefile)

    gs_class_gmm = []
    is_class_gmm = []

    median_vels_gmm = np.zeros(num_clusters)
    max_vels_gmm = np.zeros(num_clusters)
    min_vels_gmm = np.zeros(num_clusters)
    median_wids_gmm = np.zeros(num_clusters)
    max_wids_gmm = np.zeros(num_clusters)
    min_wids_gmm = np.zeros(num_clusters)

    for i in range(num_clusters):
        median_vels_gmm[i] = np.median(np.abs(vel[cluster_membership == i]))
        max_vels_gmm[i] = np.max(np.abs(vel[cluster_membership == i]))
        min_vels_gmm[i] = np.min(np.abs(vel[cluster_membership == i]))
        median_wids_gmm[i] = np.median(wid[cluster_membership == i])
        max_wids_gmm[i] = np.max(wid[cluster_membership == i])
        min_wids_gmm[i] = np.min(wid[cluster_membership == i])

        if median_vels_gmm[i] > 10:
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

    """ Calculate GMM GS/IS classification """
    num_true_gmm_gs = len(np.where((gs_flg_gmm[remove_close_range] == 1) & (emp_gs_flg[remove_close_range] == 1))[0])
    num_true_gmm_is = len(np.where((gs_flg_gmm[remove_close_range] == 0) & (emp_gs_flg[remove_close_range] == 0))[0])
    accur_gmm = float(num_true_gmm_gs+num_true_gmm_is)/num_emp*100.
    print('The GS/IS identification accurary of {}-cluster GMM is {:3.2f}%'.format(num_clusters, accur_gmm))
    print()

    """ Plot Empirical """
    cm = plt.cm.get_cmap('coolwarm')
    alpha = 0.15
    size = 1
    marker = 's'
    plot_number = 0

    plt.figure(figsize=(15, 6))
    ax0 = plt.subplot(111)
    plt.scatter(emp_time[emp_gs_flg == 0], emp_gate[emp_gs_flg == 0], s=size, c='red', marker=marker, alpha=alpha,
                cmap=cm, label='GS')  # plot IS as red
    plt.scatter(emp_time[emp_gs_flg == 1], emp_gate[emp_gs_flg == 1], s=size, c='blue', marker=marker, alpha=alpha,
                cmap=cm, label='IS')  # plot GS as blue
    ax0.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax0.set_xlim([start_time, end_time])
    ax0.set_ylim([0, 75])
    ax0.set_ylabel('Range gate')
    ax0.set_title('Empirical Model Results based on Burrell et al. 2015')

    # Print some statistics
    # First arg is left-right 0-1, Second arg is down-up 0-1
    # TODO maybe add stats for empirical for GS and IS
    plt.gcf().text(0.9, 0.8, "empirical", fontsize=12)
    plt.subplots_adjust(right=0.85, left=0.1)

    blue = mpatches.Patch(color='blue', label='ground scatter')
    red = mpatches.Patch(color='red', label='ionospheric scatter')
    plt.legend(handles=[blue, red])

    if save:
        plt.savefig(str(plot_number) + "_GMM_is_gs_" + start_time.__str__() + ".png")
        plt.close()
    if show:
        plt.show()


    """ Plot Individual Clusters """
    # Do color coding by cluster
    # TODO https://stackoverflow.com/questions/19064772/visualization-of-scatter-plots-with-overlapping-points-in-matplotlib
    #cluster_col = plt.cm.viridis(np.linspace(0, 1, num_clusters))
    cluster_col = plt.cm.plasma(np.linspace(0, 1, num_clusters))
    alpha = 1
    # Plot individually
    for i in range(num_clusters):
        plt.figure(figsize=(15, 6))
        ax0 = plt.subplot(111)
        plt.scatter(data_time[clusters[i]], gate[clusters[i]],
                    s=size, c=cluster_col[i], marker=marker, alpha=alpha, label=median_vels_gmm[i])
        ax0.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax0.set_xlim([start_time, end_time])
        ax0.set_ylim([0, 75])
        ax0.set_ylabel('Range gate')

        ax0.set_title('Gaussian Mixture Model Cluster' + (i+1).__str__())
        # Print some statistics
        stats = "Velocity:\n median {:3.1f}\n max {:3.1f}\n min {:3.1f}\n".format(
            median_vels_gmm[i], max_vels_gmm[i], min_vels_gmm[i])
        stats += "Spectral width:\n median {:3.1f}\n max {:3.1f}\n min {:3.1f}\n".format(
            median_wids_gmm[i], max_wids_gmm[i], min_wids_gmm[i])
        # First arg is left-right 0-1, Second arg is down-up 0-1
        plt.gcf().text(0.9, 0.5, stats, fontsize=10)
        plt.subplots_adjust(right=0.85, left=0.1)
        if save:
            plt.savefig((i+1).__str__() + "_GMM_cluster_" + start_time.__str__() + ".png")
            plt.close()
        if show:
            plt.show()


    """ Plot All Clusters """
    alpha = 0.15
    plot_number = num_clusters + 1
    plt.figure(figsize=(15, 6))
    ax0 = plt.subplot(111)
    ax0.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax0.set_xlim([start_time, end_time])
    ax0.set_ylabel('Range gate')
    ax0.set_title('Gaussian Mixture Model All Clusters')
    cluster_labels = [("GS" if mvel < 10 else "IS") for mvel in median_vels_gmm]
    legend_handles = []
    for i in range(num_clusters):
        plt.scatter(data_time[clusters[i]], gate[clusters[i]], s=size, c=cluster_col[i],
                    marker=marker, alpha=alpha, label=cluster_labels[i])
        legend_handles.append(mpatches.Patch(color=cluster_col[i], label=cluster_labels[i]))

    plt.legend(handles=legend_handles)
    if save:
        plt.savefig(plot_number.__str__() + "_GMM_all_clusters_" + start_time.__str__() + ".png")
        plt.close()
    if show:
        plt.show()

    """ Plot RTI """
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

    cmap, norm, bounds = utilities.genCmap(plot_param, [0, len(clusters)],
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
    cb = utilities.drawCB(fig, colormesh, cmap, norm, map_plot=0,
                      pos=[pos[0] + pos[2] + .02, pos[1], 0.02, pos[3]])

    #if show:
    plt.show()
    if save:
        plt.savefig((num_clusters + 2).__str__() + "_GMM_all_clusters_colormesh_" + start_time.__str__() + ".png")
        plt.close()


    """ Plot Histograms 
    data_flat_columns = ['beam', 'gate', 'vel', 'wid', 'power', 'phi0', 'time']
    gate = gate ** 2      # RG = RG^2
    wid = np.sign(wid) * np.log(np.abs(wid))
    vel = np.sign(vel) * np.log(np.abs(vel))
    power = np.abs(power) ** 1.5
    data_flat_unscaled = np.column_stack((beam, gate, vel, wid, power, phi0, data_time))
    data_flat_unscaled = data_flat_unscaled[remove_close_range, :]
    gs_data = data_flat_unscaled[gs_flg_gmm[remove_close_range] == 1]
    gs_data[:,2] = np.abs(gs_data[:,2])
    is_data = data_flat_unscaled[gs_flg_gmm[remove_close_range] == 0]
    is_data[:,2] = np.abs(is_data[:,2])
    plot_number = 8

    # Plot a separate histogram for each feature
    for i in range(data_flat.shape[1]):
        plt.figure(figsize=(15, 6))
        ax0 = plt.subplot(111)
        ax0.set_xlabel(data_flat_columns[i])
        ax0.set_ylabel('pdf')
        ax0.set_title('probability density for ' + data_flat_columns[i])

        gs_num_bins = len(np.unique(gs_data[:, i]))
        is_num_bins = len(np.unique(is_data[:, i]))
        if gs_num_bins > 100:
            gs_num_bins = 300
        if is_num_bins > 100:
            is_num_bins = 300

        # Use higher resolution for velocity
        gs_y, gs_binedges = np.histogram(gs_data[:, i], bins=gs_num_bins)
        is_y, is_binedges = np.histogram(is_data[:, i], bins=is_num_bins)
        gs_bincenters = 0.5 * (gs_binedges[1:] + gs_binedges[:-1])
        is_bincenters = 0.5 * (is_binedges[1:] + is_binedges[:-1])
        gs_pdf = gs_y / float(len(gs_data))
        is_pdf = is_y / float(len(is_data))

        #gs_threshhold = gs_pdf > 0.0008
        #is_threshhold = is_pdf > 0.0008
        plt.plot(gs_bincenters, gs_pdf, 'b', label='GS')
        plt.plot(is_bincenters, is_pdf, 'r', label='IS')
        #plt.xlim(xmin=0)
        #plt.xlim(xmax=650)
        #plt.ylim(ymin=0)
        plot_number += 1
        plt.legend()

        #legend_handles.append(mpatches.Patch(color=cluster_col[i], label=cluster_labels[i]))
        #plt.legend(handles=legend_handles)
        if show or i == 2 or i == 3:
            plt.show()
        if save:
            plt.savefig(str(plot_number) + '_' + data_flat_columns[i] +"_GMM_histogram" + start_time.__str__() + ".png")
            plt.close()
    """


if __name__ == '__main__':
    skip = []
    start_time = dt.datetime(2018, 2, 7)
    rad = 'cvw'
    db_path = "./Data/cvw_GSoC_2018-02-07.db"
    transform = False

    for i in range(1):
        if i in skip:
            continue

        s = start_time + dt.timedelta(i)
        e = start_time + dt.timedelta(i + 1)

        data = read_db(db_path, rad, s, e, beam=12)
        if not data:
            print('No data found')
            continue
        plot_gmm_clusters(data, s, e, num_clusters=6, show=False, save=True)