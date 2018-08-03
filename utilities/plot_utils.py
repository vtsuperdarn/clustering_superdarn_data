# TODO rename this to 'plot_utils' and non-plot stuff elsewhere


def genCmap(param, scale, colors='lasse', lowGray=False):
    """From DavitPy https://github.com/vtsuperdarn/davitpy

    Generates a colormap and returns the necessary components to use it

    Parameters
    ----------
    param : str
        the parameter being plotted ('velocity' and 'phi0' are special cases,
        anything else gets the same color scale)
    scale : list
        a list with the [min,max] values of the color scale
    colors : Optional[str]
        a string indicating which colorbar to use, valid inputs are
        'lasse', 'aj'.  default = 'lasse'
    lowGray : Optional[boolean]
        a flag indicating whether to plot low velocities (|v| < 15 m/s) in
        gray.  default = False

    Returns
    -------
    cmap : matplotlib.colors.ListedColormap
        the colormap generated.  This then gets passed to the mpl plotting
        function (e.g. scatter, plot, LineCollection, etc.)
    norm : matplotlib.colors.BoundaryNorm
        the colormap index.  This then gets passed to the mpl plotting
        function (e.g. scatter, plot, LineCollection, etc.)
    bounds : list
        the boundaries of each of the colormap segments.  This can be used
        to manually label the colorbar, for example.

    Example
    -------
        cmap,norm,bounds = genCmap('velocity', [-200,200], colors='aj', lowGray=True)

    Written by AJ 20120820

    """
    import matplotlib,numpy
    import matplotlib.pyplot as plot

    #the MPL colormaps we will be using

    cmj = matplotlib.cm.jet
    cmpr = matplotlib.cm.prism


    if(param == 'velocity'):
        #check for what color scale we want to use
        if(colors == 'aj'):
            if(not lowGray):
                #define our discrete colorbar
                cmap = matplotlib.colors.ListedColormap([cmpr(.142), cmpr(.125),
                                                         cmpr(.11), cmpr(.1),
                                                         cmpr(.175), cmpr(.158),
                                                         cmj(.32), cmj(.37)])
            else:
                cmap = matplotlib.colors.ListedColormap([cmpr(.142), cmpr(.125),
                                                         cmpr(.11), cmpr(.1),
                                                         '.6', cmpr(.175),
                                                         cmpr(.158), cmj(.32),
                                                         cmj(.37)])
        else:
            if(not lowGray):
                #define our discrete colorbar
                cmap = matplotlib.colors.ListedColormap([cmj(.9), cmj(.8),
                                                         cmj(.7), cmj(.65),
                                                         cmpr(.142), cmj(.45),
                                                         cmj(.3), cmj(.1)])
            else:
                cmap = matplotlib.colors.ListedColormap([cmj(.9), cmj(.8),
                                                         cmj(.7), cmj(.65),
                                                         '.6', cmpr(.142),
                                                         cmj(.45), cmj(.3),
                                                         cmj(.1)])

        #define the boundaries for color assignments
        bounds = numpy.round(numpy.linspace(scale[0],scale[1],7))
        if(lowGray):
            bounds[3] = -15.
            bounds = numpy.insert(bounds,4,15.)
        bounds = numpy.insert(bounds,0,-50000.)
        bounds = numpy.append(bounds,50000.)
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    elif(param == 'phi0'):
        #check for what color scale we want to use
        if(colors == 'aj'):
            #define our discrete colorbar
            cmap = matplotlib.colors.ListedColormap([cmpr(.142), cmpr(.125),
                                                     cmpr(.11), cmpr(.1),
                                                     cmpr(.18), cmpr(.16),
                                                     cmj(.32), cmj(.37)])
        else:
            #define our discrete colorbar
            cmap = matplotlib.colors.ListedColormap([cmj(.9), cmj(.8), cmj(.7),
                                                     cmj(.65), cmpr(.142),
                                                     cmj(.45), cmj(.3),
                                                     cmj(.1)])

        #define the boundaries for color assignments
        bounds = numpy.linspace(scale[0],scale[1],9)
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    elif(param == 'grid'):
        #check what color scale we want to use
        if(colors == 'aj'):
            #define our discrete colorbar
            cmap = matplotlib.colors.ListedColormap([cmpr(.175), cmpr(.17),
                                                     cmj(.32), cmj(.37),
                                                     cmpr(.142), cmpr(.13),
                                                     cmpr(.11), cmpr(.10)])
        else:
            #define our discrete colorbar
            cmap = matplotlib.colors.ListedColormap([cmj(.1), cmj(.3), cmj(.45),
                                                     cmpr(.142), cmj(.65),
                                                     cmj(.7), cmj(.8), cmj(.9)])

        #define the boundaries for color assignments
        bounds = numpy.round(numpy.linspace(scale[0],scale[1],8))
        bounds = numpy.append(bounds,50000.)
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    else:
        # If its a non-velocity plot, check what color scale we want to use
        if(colors == 'aj'):
            #define our discrete colorbar
            cmap = matplotlib.colors.ListedColormap([cmpr(.175), cmpr(.158),
                                                     cmj(.32), cmj(.37),
                                                     cmpr(.142), cmpr(.13),
                                                     cmpr(.11), cmpr(.10)])
        else:
            #define our discrete colorbar
            cmap = matplotlib.colors.ListedColormap([cmj(.1), cmj(.3), cmj(.45),
                                                     cmpr(.142), cmj(.65),
                                                     cmj(.7), cmj(.8), cmj(.9)])

        #define the boundaries for color assignments
        bounds = numpy.round(numpy.linspace(scale[0],scale[1],8))
        bounds = numpy.append(bounds,50000.)
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    cmap.set_bad('w',1.0)
    cmap.set_over('w',1.0)
    cmap.set_under('.6',1.0)

    return cmap,norm,bounds

def drawCB(fig, coll, cmap, norm, map_plot=False, pos=[0,0,1,1]):
    """ From DavitPy https://github.com/vtsuperdarn/davitpy

    manually draws a colorbar on a figure.  This can be used in lieu of
    the standard mpl colorbar function if you need the colorbar in a specific
    location.  See :func:`pydarn.plotting.rti.plotRti` for an example of its
    use.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        the figure being drawn on.
    coll : matplotlib.collections.Collection
        the collection using this colorbar
    cmap : matplotlib.colors.ListedColormap
        the colormap being used
    norm : matplotlib.colors.BoundaryNorm
        the colormap index being used
    map_plot : Optional[bool]
        a flag indicating the we are drawing the colorbar on a figure with
        a map plot
    pos : Optional[list]
        the position of the colorbar.  format = [left,bottom,width,height]

    Returns
    -------
    cb

    Example
    -------

    Written by AJ 20120820

    """
    import matplotlib.pyplot as plot

    if not map_plot:
        # create a new axes for the colorbar
        cax = fig.add_axes(pos)
        # set the colormap and boundaries for the collection of plotted items
        if(isinstance(coll,list)):
            for c in coll:
                c.set_cmap(cmap)
                c.set_norm(norm)
                cb = plot.colorbar(c,cax=cax,drawedges=True)
        else:
            coll.set_cmap(cmap)
            coll.set_norm(norm)
            cb = plot.colorbar(coll,cax=cax,drawedges=True)
    else:
        if(isinstance(coll,list)):
            for c in coll:
                c.set_cmap(cmap)
                c.set_norm(norm)
                cb = fig.colorbar(c,location='right',drawedges=True)
        else:
            coll.set_cmap(cmap)
            coll.set_norm(norm)
            cb = fig.colorbar(coll,location='right',pad="5%",drawedges=True)

    cb.ax.tick_params(axis='y',direction='out')
    return cb


#TODO why does this appear twice?
"""
def plot_clusters_colormesh(x, x_name, y, y_name, cluster_membership):
    
    Plot x vs. y, color-coded by cluster using a color mesh
    :param x:
    :param x_name:
    :param y:
    :param y_name:
    :param cluster_membership: list of integer cluster memberships. len(x) = len(y) = len(cluster_membership)
    :return:
    
    import matplotlib.pyplot as plt
    import numpy as np

    y_size = len(np.unique(y))
    y_min = np.min(y)
    y_max = np.max(y)

    x_size = len(np.unique(x))
    x_min = np.min(x)
    x_max = np.max(x)

    color_mesh = np.zeros((x_size, y_size)) * np.nan
    plot_param = 'velocity'

    # Create a (num times) x (num range gates) map of cluster values.
    # The colormap will then plot those values as cluster values.
    # cluster_color = np.linspace(-200, 200, num_clusters)
    for k in range(len(cluster_membership)):
        color = k
        # Cluster membership indices correspond to the flattened data, which may contain repeat time values
        for i in cluster_membership[k]:
            ii = np.where(x == x[i])[0][0]
            matching_y = y[ii]

            for my in matching_y:
                color_mesh[ii, my] = color

    # Create a matrix of the right size
    mesh_x, mesh_y = np.meshgrid(np.linspace(x_min, x_max, x_size), np.linspace(y_min, y_max, y_size))
    invalid_data = np.ma.masked_where(np.isnan(color_mesh.T), color_mesh.T)
    #Zm = np.ma.masked_where(np.isnan(data[:tcnt][:].T), data[:tcnt][:].T)
    # Set colormap so that masked data (bad) is transparent.

    cmap, norm, bounds = genCmap(plot_param, [0, len(cluster_membership)],
                                                 colors = 'lasse',
                                                 lowGray = False)
    cmap.set_bad('w', alpha=0.0)

    pos=[.1, .1, .76, .72]
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_axes(pos)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    #ax.set_title(title_str+' '+start_time.strftime("%d %b %Y") + ' ' + rad.upper())
    colormesh = ax.pcolormesh(mesh_x, mesh_y, invalid_data, lw=0.01, edgecolors='None', cmap=cmap, norm=norm)

    # Draw the colorbar.
    cb = drawCB(fig, colormesh, cmap, norm, map_plot=0,
                      pos=[pos[0] + pos[2] + .02, pos[1], 0.02, pos[3]])
"""

def plot_is_gs_scatterplot(time, gate, gs_flg, title):
    """
    Plot IS and GS scatterplot
    :param time:
    :param gate:
    :param gs_flg:
    :param title:
    :return:
    """
    from matplotlib.dates import DateFormatter
    import matplotlib.pyplot as plt

    cm = plt.cm.get_cmap('coolwarm')
    alpha = 0.2
    size = 1
    marker = 's'
    fig = plt.figure(figsize=(6,6))

    plt.scatter(time[gs_flg == 0], gate[gs_flg == 0],s=size,c='red',marker=marker, alpha=alpha, cmap=cm)  #plot IS as red
    plt.scatter(time[gs_flg == 1], gate[gs_flg == 1],s=size,c='blue',marker=marker, alpha=alpha, cmap=cm) #plot GS as blue
    #plt.scatter(emp_time[emp_gs_flg == -1], emp_gate[emp_gs_flg == -1],s=size,c='blue',marker=marker, alpha=alpha)  #plot the undertermined scatter as blue
    ax=plt.gca()
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    #ax1.set_xlabel('Time UT')
    ax.set_ylabel('Range gate')
    ax.set_title(title)
    # TODO get rid of this here
    plt.savefig(title + ".png")

def plot_is_gs_colormesh(ax, unique_time, time_flat, gate, gs_flg, num_range_gates, plot_closerange=False, plot_indeterminate=False):
    """
    :param ax: matplotlib axis to draw on
    :param unique_time: all the unique times from a scan
                        [date2num(d) for d in data_dict['datetime']]
                        Don't filter this to match scatter from time_flat, gate, gs_flg! Plot will not look right.
    :param time_flat: non-unique times
    :param gate:
    :param gs_flg:
    :param num_range_gates:
    :param plot_closerange:
    :param plot_indeterminate:
    :return:
    """
    #from matplotlib.dates import date2num
    import numpy as np
    import matplotlib as mpl
    from matplotlib.dates import DateFormatter

    num_times = len(unique_time)
    color_mesh = np.zeros((num_times, num_range_gates)) * np.nan

    # For IS (0) and GS (1)
    colors = [1, 2, 3]    #Will make IS (label=0) red and GS (label=1) blue
    for label in [0, 1, -1]:
        i_match = np.where(gs_flg == label)[0]
        for i in i_match:
            t = np.where(time_flat[i] == unique_time)[0][0]      # One timestamp, multiple gates.
            g = int(gate[i])
            if g <= 10 and not plot_closerange:
                continue
            if plot_indeterminate and label == -1:
                color_mesh[t, g] = colors[2]
            else:
                color_mesh[t, g] = colors[np.abs(label)]


    # Create a matrix of the right size
    range_gate = np.linspace(1, num_range_gates, num_range_gates)
    mesh_x, mesh_y = np.meshgrid(unique_time, range_gate)
    invalid_data = np.ma.masked_where(np.isnan(color_mesh.T), color_mesh.T)

    #cmap, norm, bounds = utilities.genCmap('is-gs', [0, 3], colors='lasse', lowGray=False)
    cmap = mpl.colors.ListedColormap([(1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 0.0, 1.0)])
    bounds = np.round(np.linspace(colors[0], colors[2], 3))
    bounds = np.insert(bounds, 0, -50000.)
    bounds = np.append(bounds, 50000.)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cmap.set_bad('w', alpha=0.0)

    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color='red', label='IS'), mpatches.Patch(color='blue', label='GS')]
    ax.legend(handles=handles, loc=4) # default value for loc is 'optimal', which is EXTREMELY SLOW for this plot

    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax.set_xlabel('UT')
    ax.set_xlim([unique_time[0], unique_time[-1]])
    ax.set_ylabel('Range gate')
    ax.pcolormesh(mesh_x, mesh_y, invalid_data, lw=0.01, edgecolors='None', cmap=cmap, norm=norm)


def add_colorbar(fig, bounds, colormap, pos=[0.925, 0.11, 0.015, 0.225], label=''):
    """
    :param fig:
    :param bounds:
    :param colormap:
    :param pos: [left, bottom, width, height] values are 0.0 - 1.0
    :param label:
    :return:
    """
    import matplotlib as mpl
    cax = fig.add_axes(pos)   # this list defines (left, bottom, width, height)
    norm = mpl.colors.BoundaryNorm(bounds, colormap.N)
    cb2 = mpl.colorbar.ColorbarBase(cax, cmap=colormap,
                                    norm=norm,
                                    ticks=bounds,
                                    spacing='uniform',
                                    orientation='vertical')
    cb2.set_label(label)

def plot_clusters_colormesh(ax, unique_time, time_flat, gate, clust_range, clust_labels, num_range_gates):
    import numpy as np
    import matplotlib as mpl
    from matplotlib.dates import DateFormatter
    import matplotlib.pyplot as plt

    num_times = len(unique_time)
    color_mesh = np.zeros((num_times, num_range_gates)) * np.nan

    cmap = plt.cm.jet #gist_ncar     # even more shade/color variations than Jet looks like
    unique_labels = np.unique(clust_labels)

    for c in np.unique(unique_labels):
        clust_mask = c == clust_labels
        t = [np.where(tf == unique_time)[0][0] for tf in time_flat[clust_mask]]
        g = gate[clust_mask].astype(int)
        color_mesh[t, g] = c

    # Create a matrix of the right size
    range_gate = np.linspace(1, num_range_gates, num_range_gates)
    mesh_x, mesh_y = np.meshgrid(unique_time, range_gate)
    invalid_data = np.ma.masked_where(np.isnan(color_mesh.T), color_mesh.T)

    norm = mpl.colors.BoundaryNorm(clust_range, cmap.N)
    cmap.set_bad('w', alpha=0.0)                        #TODO maybe I can get this to be black for noise
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax.set_xlabel('UT')
    ax.set_xlim([unique_time[0], unique_time[-1]])
    ax.set_ylabel('Range gate')
    ax.pcolormesh(mesh_x, mesh_y, invalid_data, lw=0.01, edgecolors='None', cmap=cmap, norm=norm)


#TODO making this and related functions into a class might simplify all this code and allow for more customization
def plot_vel_colormesh(fig, ax, unique_time, time_flat, gate, vel, num_range_gates):
    import numpy as np
    import matplotlib as mpl
    from matplotlib.dates import DateFormatter
    import matplotlib.pyplot as plt

    num_times = len(unique_time)
    color_mesh = np.zeros((num_times, num_range_gates)) * np.nan

    vel_step = 10
    vel_ranges = list(range(-100, 101, vel_step))
    vel_ranges.insert(0, -9999)
    vel_ranges.append(9999)
    cmap = plt.cm.jet  # use 'viridis' to make this redgreen colorblind proof
    for s in range(len(vel_ranges) - 1):
        step_mask = (vel >= vel_ranges[s]) & (vel < (vel_ranges[s + 1]))
        t = [np.where(tf == unique_time)[0][0] for tf in time_flat[step_mask]]
        g = gate[step_mask].astype(int)
        color_mesh[t, g] = (vel_ranges[s] + 0.5 * vel_step)

    # Create a matrix of the right size
    range_gate = np.linspace(1, num_range_gates, num_range_gates)
    mesh_x, mesh_y = np.meshgrid(unique_time, range_gate)
    invalid_data = np.ma.masked_where(np.isnan(color_mesh.T), color_mesh.T)

    norm = mpl.colors.BoundaryNorm(vel_ranges, cmap.N)
    cmap.set_bad('w', alpha=0.0)
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax.set_xlabel('UT')
    ax.set_xlim([unique_time[0], unique_time[-1]])
    ax.set_ylabel('Range gate')
    ax.pcolormesh(mesh_x, mesh_y, invalid_data, lw=0.01, edgecolors='None', cmap=cmap, norm=norm)

    # TODO this assumes ax is in the 313 position, and places the colorbar next to it. May not always be the case.
    # Would be easier to manage this if it were a class.
    add_colorbar(fig, vel_ranges, cmap, label='Vel')

def plot_stats_table(ax, stats_data_dict):
    """
    Plot statistics for a dataset on a Matplotlib subplot
    :param ax: the axis to put the stats table on - make it wide enough to accomidate all the features in stats_data_dict
    :param stats_data_dict: the dataset, in the format {'feature name' : [val, val, val, ...], ...}
    """
    import numpy as np

    feature_names = []
    feature_values = []
    for key, vals in stats_data_dict.items():
        feature_names.append(key)
        var = np.var(vals)
        med = np.median(vals)
        mean = np.mean(vals)
        max = np.max(vals)
        min = np.min(vals)
        feature_values.append([var, med, mean, max, min])

    feature_values = np.array(feature_values)       # Make sure this has the right dimensions, might need to .T
    rowLabels = ['var', 'med', 'mean', 'max', 'min']

    ax.axis('off')
    ax.table(cellText=feature_values, rowLabels=rowLabels, colLabels=feature_names, loc='center')


def make_ellipses(model, ax, n_cluster, f1, f2):
    """
    Plot an ellipse representing one cluster in GMM.

    Note: This ellipse will be centered at the mean, and its size represents the VARIANCE,
    not the STANDARD DEVIATION. Variance is taken from the diagonal of the covariance matrix.

    :param model: a GMM model trained on some data
    :param ax: the subplot axis to draw on
    :param colors: a list of colors (length num_clusters)
    :param n_cluster: integer cluster index, in range [0, num_clousters]
    :param f1: feature 1 index, in range [0, num_features]
    :param f2: feature 2 index, in range [0, num_features]
    """
    import numpy as np
    import matplotlib as mpl
    if model.covariance_type == 'full':
        covariances = model.covariances_[n_cluster][[f1, f2], :][:, [f1, f2]]
    elif model.covariance_type == 'tied':
        covariances = model.covariances_[[f1, f2], :][:, [f1, f2]]
    # TODO this may or may not work
    elif model.covariance_type == 'diag':
        covariances = np.diag(model.covariances_[n_cluster][[f1, f2]])
    # TODO will this work...? do we need it?
    elif model.covariance_type == 'spherical':
        covariances = []  # np.eye(model.means_.shape[1]) * model.covariances_[n_cluster]

    v, w = np.linalg.eigh(covariances)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    means = model.means_[n_cluster, [f1, f2]]

    ell = mpl.patches.Ellipse(means, v[0], v[1],
                              180 + angle, color='black')
    ell.set_clip_box(ax.bbox)
    ell.set_alpha(0.5)
    ax.add_artist(ell)

def plot_feature_pairs_by_cluster(data_flat, estimator, feature_names, save=True, base_path = ''):
    import matplotlib.pyplot as plt
    import numpy as np
    from math import ceil
    """
    gate = gate ** 2      # RG = RG^2
    wid = np.sign(wid) * np.log(np.abs(wid))
    vel = np.sign(vel) * np.log(np.abs(vel))
    power = np.abs(power) ** 1.5
    """


    #gs_flg_gmm, clusters, median_vels_gmm, estimator = gmm(data_flat, vel, wid, num_clusters=num_clusters, cluster_identities=True)

    num_features = len(feature_names)
    cluster_ids = estimator.predict(data_flat)
    #velocity_ordering = np.argsort(median_vels_gmm)
    #clusters = [clusters[i] for i in velocity_ordering]
    #median_vels_gmm = [median_vels_gmm[i] for i in velocity_ordering]
    #cluster_labels = [("GS" if mvel < 15 else "IS") for mvel in median_vels_gmm]
    num_clusters = len(np.unique(cluster_ids))
    colors = plt.cm.plasma(np.linspace(0, 1, num_clusters))

    for f1 in range(num_features):
        for f2 in range(f1+1, num_features):
            plt.figure(figsize=(20, 12))
            plot_name = feature_names[f1] + " vs " + feature_names[f2]

            for c in range(num_clusters):
                #print('===========================')
                #print('cluster', c, 'features', f1, f2)
                #print('===========================')
                #print(feature_names[f1], 'mean', np.mean(data_flat[cluster_ids == c, f1]), 'var', np.var(data_flat[cluster_ids == c, f1]))
                #print(feature_names[f2], 'mean', np.mean(data_flat[cluster_ids == c, f2]), 'var', np.var(data_flat[cluster_ids == c, f2]))
                #print()

                ax = plt.subplot(2, ceil(num_clusters / 2.0), c + 1)
                ax.scatter(data_flat[cluster_ids == c, f1], data_flat[cluster_ids == c, f2],
                           alpha=0.1, marker='x', color=colors[c])

                make_ellipses(estimator, ax, c, f1, f2)

                #plt.legend()

                plt.xlabel(feature_names[f1])
                plt.ylabel((feature_names[f2]))
                plt.title(plot_name)

            if save:
                plt.savefig(base_path + plot_name + '.png')
                plt.close()
            else:
                plt.show()
