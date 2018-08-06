import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter


class RangeTimePlot(object):
    """
    Create plots for IS/GS flags, velocity, and algorithm clusters.
    """

    def __init__(self, nrang, unique_times, num_subplots=3):
        self.nrang = nrang
        self.unique_gates = np.linspace(1, nrang, nrang)
        self.unique_times = unique_times
        self.num_subplots = num_subplots
        self._num_subplots_created = 0
        self.fig = plt.figure(figsize=(14, 5*num_subplots))


    def addClusterPlot(self, data_dict, clust_flg, beam, title, show_closerange=True):
        # add new axis
        self.cluster_ax = self._add_axis()
        # set up variables for plotter
        time = np.hstack(data_dict['time'])
        gate = np.hstack(data_dict['gate'])
        allbeam = np.hstack(data_dict['beam'])
        flags = np.hstack(clust_flg)
        # TODO need to randomize colors or something to plot large # of clusters well
        # and preferably be able to show noise... altho that will be on ISGS plot
        # Randomize flags so that colors are randomized
        # (DBSCAN clusters next to each other will be plotted in different colors)
        import copy
        unique_flgs = np.unique(flags)
        clust_range = list(range(int(min(unique_flgs)), int(max(unique_flgs)) + 1))
        flg_randomizer = copy.deepcopy(clust_range)
        np.random.seed(0)
        np.random.shuffle(flg_randomizer)
        random_flags = np.zeros(len(flags))
        for f in unique_flgs:
            cluster_mask = f == flags
            if f == -1:
                random_flags[cluster_mask] = -1
            else:
                random_flags[cluster_mask] = flg_randomizer[f]

        mask = allbeam == beam
        if show_closerange:
            mask = mask & (gate > 10)
        if -1 in flags:
            cmap = plt.cm.nipy_spectral
        else:
            cmap = plt.cm.gist_rainbow      # no black or white

        # Lower bound for cmap is inclusive, upper bound is non-inclusive
        bounds = list(range(np.min(flags), np.max(flags)+2))    # need (max_cluster+1) to be the upper bound
        self._create_colormesh(self.cluster_ax, time, gate, random_flags, mask, bounds, cmap)
        self.cluster_ax.set_title(title)


    def addGSISPlot(self, data_dict, gs_flg, beam, title, show_closerange=True):
        # add new axis
        self.isgs_ax = self._add_axis()
        # set up variables for plotter
        time = np.hstack(data_dict['time'])
        gate = np.hstack(data_dict['gate'])
        allbeam = np.hstack(data_dict['beam'])
        flags = np.hstack(gs_flg)
        mask = allbeam == beam
        if show_closerange:
            mask = mask & (gate > 10)
        if -1 in flags:                     # contains noise flag
            cmap = mpl.colors.ListedColormap([(0.0, 0.0, 0.0, 1.0),     # black
                                              (1.0, 0.0, 0.0, 1.0),     # blue
                                              (0.0, 0.0, 1.0, 1.0)])    # red
            bounds = [-1, 0, 1, 2] # Lower bound inclusive, upper bound non-inclusive
        else:
            cmap = mpl.colors.ListedColormap([(1.0, 0.0, 0.0, 1.0),  # blue
                                              (0.0, 0.0, 1.0, 1.0)])  # red
            bounds = [0, 1, 2]  # Lower bound inclusive, upper bound non-inclusive
        self._create_colormesh(self.isgs_ax, time, gate, flags, mask, bounds, cmap)
        self.isgs_ax.set_title(title)


    def addVelPlot(self, data_dict, beam, title, vel_max=200, vel_step=25, show_closerange=True):
        # add new axis
        self.vel_ax = self._add_axis()
        # set up variables for plotter
        time = np.hstack(data_dict['time'])
        gate = np.hstack(data_dict['gate'])
        allbeam = np.hstack(data_dict['beam'])
        flags = np.hstack(data_dict['vel'])
        bounds = list(range(-vel_max, vel_max+1, vel_step))
        cmap = plt.cm.jet
        mask = allbeam == beam
        self._create_colormesh(self.vel_ax, time, gate, flags, mask, bounds, cmap)
        self._add_colorbar(self.vel_ax, bounds, cmap, label='Velocity')
        self.vel_ax.set_title(title)


    def show(self):
        plt.show()


    # Private helper functions

    def _add_axis(self):
        self._num_subplots_created += 1
        ax = self.fig.add_subplot(self.num_subplots, 1, self._num_subplots_created)
        return ax


    def _add_colorbar(self, ax, bounds, cmap, label=''):
        # position the colorbar to the right of the axis
        pos = ax.get_position()
        cpos = [pos.x0 + 0.8, pos.y0 + 0.0125, 0.015, pos.height * 0.9]
        # create axis for colorbar
        cax = self.fig.add_axes(cpos)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cb2 = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                        norm=norm,
                                        ticks=bounds,
                                        spacing='uniform',
                                        orientation='vertical')
        cb2.set_label(label)


    def _create_colormesh(self, ax, time, gate, flags, mask, bounds, cmap):
        # Create a (n times) x (n range gates) array and add flag data
        num_times = len(self.unique_times)
        color_mesh = np.zeros((num_times, self.nrang)) * np.nan
        for f in np.unique(flags):
            flag_mask = (flags == f) & mask
            t = [np.where(tf == self.unique_times)[0][0] for tf in time[flag_mask]]
            g = gate[flag_mask].astype(int)
            color_mesh[t, g] = f
        # Set up variables for pcolormesh
        mesh_x, mesh_y = np.meshgrid(self.unique_times, self.unique_gates)
        masked_colormesh = np.ma.masked_where(np.isnan(color_mesh.T), color_mesh.T)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        # Configure axes
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        hours = mdates.HourLocator(byhour=range(0, 24, 4))
        ax.xaxis.set_major_locator(hours)
        ax.set_xlabel('UT')
        ax.set_xlim([self.unique_times[0], self.unique_times[-1]])
        ax.set_ylabel('Range gate')
        ax.pcolormesh(mesh_x, mesh_y, masked_colormesh, lw=0.01, edgecolors='None', cmap=cmap, norm=norm)



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


def plot_feature_pairs_by_cluster(data_flat, estimator, feature_names, save=True, base_path = ''):
    import matplotlib.pyplot as plt
    import numpy as np
    from math import ceil
    """ Transformed features
    gate = gate ** 2      # RG = RG^2
    wid = np.sign(wid) * np.log(np.abs(wid))
    vel = np.sign(vel) * np.log(np.abs(vel))
    power = np.abs(power) ** 1.5
    """

    num_features = len(feature_names)
    cluster_ids = estimator.predict(data_flat)
    num_clusters = len(np.unique(cluster_ids))
    colors = plt.cm.plasma(np.linspace(0, 1, num_clusters))

    for f1 in range(num_features):
        for f2 in range(f1+1, num_features):
            plt.figure(figsize=(20, 12))
            plot_name = feature_names[f1] + " vs " + feature_names[f2]

            for c in range(num_clusters):

                ax = plt.subplot(2, ceil(num_clusters / 2.0), c + 1)
                ax.scatter(data_flat[cluster_ids == c, f1], data_flat[cluster_ids == c, f2],
                           alpha=0.1, marker='x', color=colors[c])

                _make_ellipses(estimator, ax, c, f1, f2)


                plt.xlabel(feature_names[f1])
                plt.ylabel((feature_names[f2]))
                plt.title(plot_name)

            if save:
                plt.savefig(base_path + plot_name + '.png')
                plt.close()
            else:
                plt.show()


def _make_ellipses(model, ax, n_cluster, f1, f2):
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
