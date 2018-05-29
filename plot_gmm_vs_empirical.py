from cluster import *
import utilities
import matplotlib as mpl

"""
mpl subplots: https://matplotlib.org/devdocs/api/_as_gen/matplotlib.pyplot.subplot.html#matplotlib.pyplot.subplot
"""

def gmm_vs_empirical_scatter(data_dict, start_time, end_time, clusters=6, show=True, save=False):
    """
    Compare traditional, empirical, kmeans, and GMM
    :param data_dict: dictionary from read_db
    :param start_time: datetime object
    :param end_time: datetime object
    :return: creates a graph as PNG, title includes start_time
    """
    trad_gs_flg = traditional(data_dict)
    emp_gs_flg, emp_time, emp_gate = empirical(data_dict)
    data_flat, beam, gate, vel, wid, power, phi0, time = flatten_data(data_dict, extras=True)

    """
    # kmeans_gs_flg = kmeans(data_flat, vel, wid)
    """

    remove_close_range = gate >= 10
    time = time[remove_close_range]
    gate = gate[remove_close_range]
    trad_gs_flg = trad_gs_flg[remove_close_range]
    emp_gs_flg = emp_gs_flg[remove_close_range]

    # TODO Is there some difference with emp_time and emp_gate? That may throw it off?
    #kmeans_gs_flg = kmeans_gs_flg[remove_close_range]
    gmm_gs_flg = gmm(data_flat, vel, wid, num_clusters=clusters)
    gmm_gs_flg = gmm_gs_flg[remove_close_range]

    num_true_trad_gs = len(np.where(((trad_gs_flg == 1) | (trad_gs_flg == -1)) & (emp_gs_flg == 1))[0])
    num_true_trad_is = len(np.where(((trad_gs_flg == 0)) & (emp_gs_flg == 0))[0])

    num_emp = len(emp_gs_flg)
    accur_tra = float(num_true_trad_gs+num_true_trad_is)/num_emp*100.
    print('The GS/IS identification accurary of traditional method is {:3.2f}%'.format(accur_tra))

    """
    num_true_kmeans_gs = len(np.where((kmeans_gs_flg == 1) & (emp_gs_flg == 1))[0]) #Assuming the GS is the cluster with minimum median velocity
    num_true_kmeans_is = len(np.where((kmeans_gs_flg == 0) & (emp_gs_flg == 0))[0])
    accur_kmeans = float(num_true_kmeans_gs+num_true_kmeans_is)/num_emp*100.
    print 'The GS/IS identification accurary of kmeans is {:3.2f}%'.format(accur_kmeans)
    """

    num_true_gmm_gs = len(np.where((gmm_gs_flg == 1) & (emp_gs_flg == 1))[0]) #Assuming the GS is the cluster with minimum median velocity
    num_true_gmm_is = len(np.where((gmm_gs_flg == 0) & (emp_gs_flg == 0))[0])
    accur_gmm = float(num_true_gmm_gs+num_true_gmm_is)/num_emp*100.
    print('The GS/IS identification accurary of GMM is {:3.2f}%'.format(accur_gmm))

    tran_gmm_data_flat, _ = flatten_data(data_dict, transform=True)
    tran_gmm_gs_flg = gmm(tran_gmm_data_flat, vel, wid, num_clusters=clusters)
    tran_gmm_gs_flg = tran_gmm_gs_flg[remove_close_range]
    num_true_tran_gmm_gs = len(np.where((tran_gmm_gs_flg == 1) & (emp_gs_flg == 1))[0]) #Assuming the GS is the cluster with minimum median velocity
    num_true_tran_gmm_is = len(np.where((tran_gmm_gs_flg == 0) & (emp_gs_flg == 0))[0])
    accur_tran_gmm = float(num_true_tran_gmm_gs+num_true_tran_gmm_is)/num_emp*100.
    print('The GS/IS identification accurary of transformed GMM is {:3.2f}%'.format(accur_tran_gmm))

    cm = plt.cm.get_cmap('coolwarm')
    alpha = 0.2
    size = 1
    marker = 's'

    fig = plt.figure(figsize=(20,8))
    ax1 = plt.subplot(411)
    plt.scatter(emp_time[emp_gs_flg == 0], emp_gate[emp_gs_flg == 0],s=size,c='red',label='IS',marker=marker, alpha=alpha, cmap=cm)  #plot IS as red
    plt.scatter(emp_time[emp_gs_flg == 1], emp_gate[emp_gs_flg == 1],s=size,c='blue',label='GS',marker=marker, alpha=alpha, cmap=cm) #plot GS as blue
    #plt.scatter(emp_time[emp_gs_flg == -1], emp_gate[emp_gs_flg == -1],s=size,c='blue',marker=marker, alpha=alpha)  #plot the undertermined scatter as blue
    ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    #ax1.set_xlabel('Time UT')
    ax1.set_xlim([start_time,end_time])
    ax1.set_ylabel('Range gate')


    ax2 = plt.subplot(412)
    plt.scatter(time[trad_gs_flg == 0], gate[trad_gs_flg == 0],s=size,c='red',marker=marker, alpha=alpha, cmap=cm)  #plot IS as red
    plt.scatter(time[trad_gs_flg == 1], gate[trad_gs_flg == 1],s=size,c='blue',marker=marker, alpha=alpha, cmap=cm) #plot GS as blue
    #the indeterminate updated gflg (-1) was original ground scatter in traditional method when using the emp_data_dict
    plt.scatter(time[trad_gs_flg == -1], gate[trad_gs_flg == -1],s=size,c='blue',marker=marker, alpha=alpha, cmap=cm)
    ax2.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax2.set_xlim([start_time,end_time])
    ax2.set_ylabel('Range gate')
    ax2.set_title('Traditional Model Results [Blanchard et al. 2009] ({:3.2f}% agree with empirical)'.format(accur_tra))

    """
    ax3 = plt.subplot(413)
    plt.scatter(time[gs_flg_kmeans == 0], gate[gs_flg_kmeans == 0],s=size,c='red',marker=marker,alpha = alpha, cmap=cm)
    plt.scatter(time[gs_flg_kmeans == 1], gate[gs_flg_kmeans == 1],s=size,c='blue',marker=marker,alpha = alpha, cmap=cm)
    ax3.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    #ax3.set_xlabel('Time UT')
    ax3.set_xlim([start_time,end_time])
    ax3.set_ylabel('Range gate')
    ax3.set_title('Kmeans Results ({:3.2f}% agree with empirical)'.format(accur_kmeans))
    """

    ax4 = plt.subplot(414)
    plt.scatter(time[gmm_gs_flg == 0], gate[gmm_gs_flg == 0],s=size,c='red',marker=marker,alpha = alpha, cmap=cm)  #plot ionospheric scatter as red
    plt.scatter(time[gmm_gs_flg == 1], gate[gmm_gs_flg == 1],s=size,c='blue',marker=marker,alpha = alpha, cmap=cm) #plot ground scatter as blue
    ax4.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax4.set_xlabel('Time UT')
    ax4.set_xlim([start_time,end_time])
    ax4.set_ylabel('Range gate')
    ax4.set_title('Gaussian Mixture Model Results ({:3.2f}% agree with empirical)'.format(accur_gmm))

    filename = "gmm vs. emp vs. empirical scatter " + start_time.__str__() + ".png"
    fig.tight_layout()
    if save:
        plt.savefig(filename)
    if show:
        plt.show()
    if save:
        plt.close()


def gmm_vs_empirical_colormesh(data_dict, start_time, end_time, clusters=6, show=True, save=False):
    """
    Compare traditional, empirical, kmeans, and GMM
    :param data_dict: dictionary from read_db
    :param start_time: datetime object
    :param end_time: datetime object
    :return: creates a graph as PNG, title includes start_time
    """
    trad_gs_flg = traditional(data_dict)
    emp_gs_flg, emp_time, emp_gate = empirical(data_dict)
    data_flat, beam, gate, vel, wid, power, phi0, time_flat = flatten_data(data_dict, extras=True)

    remove_close_range = gate >= 10
    time_flat = time_flat[remove_close_range]
    gate = gate[remove_close_range]

    # Mark indeterminate scatter in empirical (determined by negative values in the traditional GS flag)
    # TODO THIS IS GONNA AFFECT SOME STUFF should we remove it? how to deal with the 'emp agreement' calculations?
    # TODO we need to filter out ind. scatter like we do with the close range-gate
    # TODO the black and blue are hard to tell apart
    indeterminate = np.where(trad_gs_flg == -1)
    emp_gs_flg[indeterminate] = -1
    emp_gs_flg = emp_gs_flg[remove_close_range]
    num_emp = len(emp_gs_flg)

    trad_gs_flg = trad_gs_flg[remove_close_range]
    num_true_trad_gs = len(np.where(((trad_gs_flg == 1) | (trad_gs_flg == -1)) & (emp_gs_flg == 1))[0])
    num_true_trad_is = len(np.where(((trad_gs_flg == 0)) & (emp_gs_flg == 0))[0])
    accur_tra = float(num_true_trad_gs+num_true_trad_is)/num_emp*100.
    print('The GS/IS identification accurary of traditional method is {:3.2f}%'.format(accur_tra))

    gmm_gs_flg = gmm(data_flat, vel, wid, num_clusters=clusters)
    gmm_gs_flg = gmm_gs_flg[remove_close_range]
    num_true_gmm_gs = len(np.where((gmm_gs_flg == 1) & (emp_gs_flg == 1))[0]) #Assuming the GS is the cluster with minimum median velocity
    num_true_gmm_is = len(np.where((gmm_gs_flg == 0) & (emp_gs_flg == 0))[0])
    accur_gmm = float(num_true_gmm_gs+num_true_gmm_is)/num_emp*100.
    print('The GS/IS identification accurary of GMM is {:3.2f}%'.format(accur_gmm))

    tran_gmm_data_flat, _ = flatten_data(data_dict, transform=True)
    tran_gmm_gs_flg = gmm(tran_gmm_data_flat, vel, wid, num_clusters=clusters)
    tran_gmm_gs_flg = tran_gmm_gs_flg[remove_close_range]
    num_true_tran_gmm_gs = len(np.where((tran_gmm_gs_flg == 1) & (emp_gs_flg == 1))[0]) #Assuming the GS is the cluster with minimum median velocity
    num_true_tran_gmm_is = len(np.where((tran_gmm_gs_flg == 0) & (emp_gs_flg == 0))[0])
    accur_tran_gmm = float(num_true_tran_gmm_gs+num_true_tran_gmm_is)/num_emp*100.
    print('The GS/IS identification accurary of transformed GMM is {:3.2f}%'.format(accur_tran_gmm))

    num_range_gates = data_dict['nrang'][0]
    time = data_dict['datetime']

    beams = np.unique(beam)
    for b in beams:
        scatter_notflat = b == np.array(data_dict['beam'])[np.array(data_dict['gate']) > 10]
        scatter_flat = b == beam[remove_close_range]
        time = np.array(time)

        fig = plt.figure(figsize=(20,8))
        ax1 = plt.subplot(411)
        plot_rti(ax1, time[scatter_notflat], time_flat[scatter_flat], gate[scatter_flat], emp_gs_flg[scatter_flat], num_range_gates, plot_indeterminate=True)
        ax1.set_title('Empirical Model Results [Burrell et al. 2015] Beam ' + int(b).__str__())
        ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax1.set_xlim([start_time, end_time])

        ax2 = plt.subplot(412)
        plot_rti(ax2, time[scatter_notflat], time_flat[scatter_flat], gate[scatter_flat], trad_gs_flg[scatter_flat], num_range_gates)
        ax2.set_title('Traditional Model Results [Blanchard et al. 2009] ({:3.2f}% agree with empirical)'.format(accur_tra))
        ax2.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax2.set_xlim([start_time, end_time])

        ax3 = plt.subplot(413)
        plot_rti(ax3, time[scatter_notflat], time_flat[scatter_flat], gate[scatter_flat], gmm_gs_flg[scatter_flat], num_range_gates)
        ax3.set_title('Gaussian Mixture Model Results ({:3.2f}% agree with empirical)'.format(accur_gmm))
        ax3.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax3.set_xlim([start_time, end_time])

        ax4 = plt.subplot(414)
        plot_rti(ax4, time[scatter_notflat], time_flat[scatter_flat], gate[scatter_flat], tran_gmm_gs_flg[scatter_flat], num_range_gates)
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


def plot_rti(ax, time, time_flat, gate, gs_flg, num_range_gates, plot_indeterminate=False):
    time = date2num(time)
    num_times = len(time)
    color_mesh = np.zeros((num_times, num_range_gates)) * np.nan
    plot_param = 'velocity'

    # For IS (0) and GS (1)
    colors = [1,2,3]    #Will make IS (label=0) red and GS (label=1) blue
    for label in [0, 1, -1]:
        i_match = np.where(gs_flg == label)[0]
        for i in i_match:
            t = np.where(time_flat[i] == time)[0][0]      # One timestamp, multiple gates.
            g = gate[i]
            if plot_indeterminate and label == -1:
                color_mesh[t, g] = colors[2]
            else:
                color_mesh[t, g] = colors[np.abs(label)]


    # Create a matrix of the right size
    range_gate = np.linspace(1, num_range_gates, num_range_gates)
    mesh_x, mesh_y = np.meshgrid(time, range_gate)
    invalid_data = np.ma.masked_where(np.isnan(color_mesh.T), color_mesh.T)

    cmap, norm, bounds = utilities.genCmap('is-gs', [0, 3], colors='lasse', lowGray=False)
    cmap = mpl.colors.ListedColormap([(1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 0.0, 1.0)])
    bounds = np.round(np.linspace(colors[0], colors[2], 3))
    bounds = np.insert(bounds, 0, -50000.)
    bounds = np.append(bounds, 50000.)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cmap.set_bad('w', alpha=0.0)

    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax.set_xlabel('UT')
    ax.set_xlim([time[0], time[-1]])
    ax.set_ylabel('Range gate')
    ax.pcolormesh(mesh_x, mesh_y, invalid_data, lw=0.01, edgecolors='None', cmap=cmap, norm=norm)


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
        gmm_vs_empirical_colormesh(data, s, e, clusters=3, show=False, save=True)