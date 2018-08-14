import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter, num2date
from matplotlib import patches
import matplotlib.patches as mpatches
import random


CLUSTER_CMAP = plt.cm.gist_rainbow

def get_cluster_cmap():
    cmap = CLUSTER_CMAP
    cmaplist = [cmap(i) for i in range(cmap.N)]
    random.seed(10)
    random.shuffle(cmaplist)
    return cmap.from_list('Cluster cmap', cmaplist, cmap.N)


def get_cluster_noise_cmap():
    cmap = CLUSTER_CMAP
    cmaplist = np.array([cmap(i) for i in range(cmap.N)])
    random.seed(10)
    random.shuffle(cmaplist)
    cmaplist[0] = (0, 0, 0, 1.0)
    return cmap.from_list('Cluster cmap', cmaplist, cmap.N)


class RangeTimePlot(object):
    """
    Create plots for IS/GS flags, velocity, and algorithm clusters.
    """
    def __init__(self, nrang, unique_times, fig_title, num_subplots=3):
        self.nrang = nrang
        self.unique_gates = np.linspace(1, nrang, nrang)
        self.unique_times = unique_times
        self.num_subplots = num_subplots
        self._num_subplots_created = 0
        self.fig = plt.figure(figsize=(8, 3*num_subplots), dpi=100) # Size for website
        plt.suptitle(fig_title, x=0.075, y=0.99, ha='left', fontweight='bold', fontsize=15)
        mpl.rcParams.update({'font.size': 10})



    def addClusterPlot(self, data_dict, clust_flg, beam, title, show_closerange=True, xlabel=''):
        # add new axis
        self.cluster_ax = self._add_axis()
        # set up variables for plotter
        time = np.hstack(data_dict['time'])
        gate = np.hstack(data_dict['gate'])
        allbeam = np.hstack(data_dict['beam'])
        flags = np.hstack(clust_flg)
        mask = allbeam == beam
        if not show_closerange:
            mask = mask & (gate > 10)
        if -1 in flags:
            cmap = get_cluster_noise_cmap()       # black for noise
        else:
            cmap = get_cluster_cmap()             # no black or grey

        # Lower bound for cmap is inclusive, upper bound is non-inclusive
        bounds = list(range(int(np.min(flags)), int(np.max(flags))+2))    # need (max_cluster+1) to be the upper bound
        self._create_colormesh(self.cluster_ax, time, gate, flags, mask, bounds, cmap, xlabel)
        self.cluster_ax.set_title(title,  loc='left', fontdict={'fontweight': 'bold'})


    def addGSISPlot(self, data_dict, gs_flg, beam, title, show_closerange=True, xlabel=''):
        # add new axis
        self.isgs_ax = self._add_axis()
        # set up variables for plotter
        time = np.hstack(data_dict['time'])
        gate = np.hstack(data_dict['gate'])
        allbeam = np.hstack(data_dict['beam'])
        flags = np.hstack(gs_flg)
        mask = allbeam == beam
        if not show_closerange:
            mask = mask & (gate > 10)
        if -1 in flags:                     # contains noise flag
            cmap = mpl.colors.ListedColormap([(0.0, 0.0, 0.0, 1.0),     # black
                                              (1.0, 0.0, 0.0, 1.0),     # blue
                                              (0.0, 0.0, 1.0, 1.0)])    # red
            bounds = [-1, 0, 1, 2]      # Lower bound inclusive, upper bound non-inclusive
        else:
            cmap = mpl.colors.ListedColormap([(1.0, 0.0, 0.0, 1.0),  # blue
                                              (0.0, 0.0, 1.0, 1.0)])  # red
            bounds = [0, 1, 2]          # Lower bound inclusive, upper bound non-inclusive
        self._create_colormesh(self.isgs_ax, time, gate, flags, mask, bounds, cmap, xlabel)
        self.isgs_ax.set_title(title,  loc='left', fontdict={'fontweight': 'bold'})
        # Add a legend
        handles = [mpatches.Patch(color='red', label='IS'), mpatches.Patch(color='blue', label='GS')]
        self.isgs_ax.legend(handles=handles, loc=4)


    def addVelPlot(self, data_dict, beam, title, vel_max=200, vel_step=25, xlabel='Time UT'):
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
        self._create_colormesh(self.vel_ax, time, gate, flags, mask, bounds, cmap, xlabel)
        self._tight_layout()    # need to do this before adding the colorbar, because it depends on the axis position
        self._add_colorbar(self.fig, self.vel_ax, bounds, cmap, label='Velocity [m/s]')
        self.vel_ax.set_title(title, loc='left', fontdict={'fontweight': 'bold'})


    def _tight_layout(self):
        self.fig.tight_layout(rect=[0, 0, 0.9, 0.97])

    def show(self):
        plt.show()


    def save(self, filepath):
        plt.savefig(filepath)


    def close(self):
        self.fig.clf()
        plt.close()

    # Private helper functions

    def _add_axis(self):
        self._num_subplots_created += 1
        ax = self.fig.add_subplot(self.num_subplots, 1, self._num_subplots_created)
        return ax

    def _add_colorbar(self, fig, ax, bounds, colormap, label=''):
        """
        Add a colorbar to the right of an axis.
        :param fig:
        :param ax:
        :param bounds:
        :param colormap:
        :param label:
        :return:
        """
        import matplotlib as mpl
        pos = ax.get_position()
        cpos = [pos.x1 + 0.025, pos.y0 + 0.0125,
                0.015, pos.height * 0.9]                # this list defines (left, bottom, width, height
        cax = fig.add_axes(cpos)
        norm = mpl.colors.BoundaryNorm(bounds, colormap.N)
        cb2 = mpl.colorbar.ColorbarBase(cax, cmap=colormap,
                                        norm=norm,
                                        ticks=bounds,
                                        spacing='uniform',
                                        orientation='vertical')
        cb2.set_label(label)


    def _create_colormesh(self, ax, time, gate, flags, mask, bounds, cmap, xlabel=''):
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
        # cmap.set_bad('w', alpha=0.0)
        # Configure axes
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        hours = mdates.HourLocator(byhour=range(0, 24, 4))
        ax.xaxis.set_major_locator(hours)
        ax.set_xlabel(xlabel)
        ax.set_xlim([self.unique_times[0], self.unique_times[-1]])
        ax.set_ylabel('Range gate')
        ax.pcolormesh(mesh_x, mesh_y, masked_colormesh, lw=0.01, edgecolors='None', cmap=cmap, norm=norm)



class FanPlot:


    def __init__(self, nrange=75, nbeam=16, r0=180, dr=45, dtheta=3.24, theta0=None):
        """
        Initialize the fanplot do a certain size.
        :param nrange: number of range gates
        :param nbeam: number of beams
        :param r0: initial beam distance - any distance unit as long as it's consistent with dr
        :param dr: length of each radar - any distance unit as long as it's consistent with r0
        :param dtheta: degrees per beam gate, degrees (default 3.24 degrees)
        """
        # Set member variables
        self.nrange = int(nrange)
        self.nbeam = int(nbeam)
        self.r0 = r0
        self.dr = dr
        self.dtheta = dtheta
        # Initial angle (from X, polar coordinates) for beam 0
        if theta0 == None:
            self.theta0 = (90 - dtheta * nbeam / 2)     # By default, point fanplot towards 90 deg
        else:
            self.theta0 = theta0


    def plot_clusters(self, data_dict, clust_flg, scans, name,
                      vel_max=200, vel_step=25,
                      show=True, save=False, base_filepath=''):
        unique_clusters = np.unique(np.hstack(clust_flg))
        if -1 in np.hstack(unique_clusters):
            cluster_cmap = get_cluster_noise_cmap()
        else:
            cluster_cmap = get_cluster_cmap()

        colors = np.array(cluster_cmap(range(cluster_cmap.N)))
                                #range(int(np.min(unique_clusters)+1), int(np.max(unique_clusters)) + 2)
                             # ))
        n = np.array(range(int(np.min(unique_clusters) + 1), int(np.max(unique_clusters)) + 1)) % cluster_cmap.N
        np.random.seed(0)
        np.random.shuffle(n)
        cluster_colors = [colors[0]]
        cluster_colors.extend(colors[n])
        vel_ranges = list(range(-vel_max, vel_max + 1, vel_step))
        vel_ranges.insert(0, -9999)
        vel_ranges.append(9999)
        vel_cmap = plt.cm.jet       # use 'viridis' colormap to make this redgreen colorblind proof
        vel_colors = vel_cmap(np.linspace(0, 1, len(vel_ranges)))

        for i in scans:
            fig = plt.figure(figsize=(16, 9))
            clust_ax = self.add_axis(fig, 121)
            clust_i = np.unique(clust_flg[i]).astype(int)
            # Cluster fanplot
            for ci, c in enumerate(clust_i):
                clust_mask = clust_flg[i] == c
                beam_c = data_dict['beam'][i][clust_mask]
                gate_c = data_dict['gate'][i][clust_mask]
                color = cluster_colors[c+1]
                if c != -1:
                    m = int(len(beam_c) / 2)  # Beam is sorted, so this is roughly the index of the median beam
                    self.text(str(c), beam_c[m], gate_c[m])  # Label cluster #
                self.plot(clust_ax, beam_c, gate_c, color)
            clust_ax.set_title('Clusters')
            # Velocity fanplot
            vel_ax = self.add_axis(fig, 122)
            for s in range(len(vel_ranges) - 1):
                step_mask = (data_dict['vel'][i] >= vel_ranges[s]) & (data_dict['vel'][i] <= (vel_ranges[s + 1]))
                beam_s = data_dict['beam'][i][step_mask]
                gate_s = data_dict['gate'][i][step_mask]
                self.plot(vel_ax, beam_s, gate_s, vel_colors[s])
            self._add_colorbar(fig, vel_ax, vel_ranges, vel_cmap, label='Velocity [m/s]')
            vel_ax.set_title('Velocity')
            # Add title
            scan_time = num2date(data_dict['time'][i][0]).strftime('%H:%M:%S')
            plt.suptitle('\n\n%sscan time %s' % (name, scan_time))
            if save:
                filepath = '%s_%s.jpg' % (base_filepath, scan_time)
                plt.savefig(filepath)
            if show:
                plt.show()
            fig.clf()
            plt.close()


    def add_axis(self, fig, subplot):
        ax = fig.add_subplot(subplot, polar=True)

        # Set up ticks and labels
        self.r_ticks = range(self.r0, self.r0 + (self.nrange+1) * self.dr, self.dr)
        self.theta_ticks = [self.theta0 + self.dtheta * b for b in range(self.nbeam+1)]
        rlabels = [""] * len(self.r_ticks)
        for i in range(0, len(rlabels), 5):
            rlabels[i] = i
        plt.rgrids(self.r_ticks, rlabels)
        plt.thetagrids(self.theta_ticks, range(self.nbeam))
        return ax


    def plot(self, ax, beams, gates, color="blue"):
        """
        Add some data to the plot in a single color at positions given by 'beams' and 'gates'.
        :param beams: a list/array of beams
        :param gates: a list/array of gates - same length as beams
        :param color: a Matplotlib color
        """
        for i, (beam, gate) in enumerate(zip(beams, gates)):
            theta = (self.theta0 + beam * self.dtheta) * np.pi / 180        # radians
            r = (self.r0 + gate * self.dr)                                  # km
            width = self.dtheta * np.pi / 180                               # radians
            height = self.dr                                                # km

            x1, x2 = theta, theta + width
            y1, y2 = r, r + height
            x = x1, x2, x2, x1
            y = y1, y1, y2, y2
            ax.fill(x, y, color=color)
        self._scale_plot(ax)


    def _add_colorbar(self, fig, ax, bounds, colormap, label=''):
        """
        Add a colorbar to the right of an axis.
        Similar to the function in RangeTimePlot, but positioned differently fanplots.
        :param fig:
        :param ax:
        :param bounds:
        :param colormap:
        :param label:
        :return:
        """
        import matplotlib as mpl
        pos = ax.get_position()
        cpos = [pos.x1 + 0.025, pos.y0 + 0.25*pos.height,
                0.01, pos.height * 0.5]            # this list defines (left, bottom, width, height
        cax = fig.add_axes(cpos)
        norm = mpl.colors.BoundaryNorm(bounds, colormap.N)
        cb2 = mpl.colorbar.ColorbarBase(cax, cmap=colormap,
                                        norm=norm,
                                        ticks=bounds,
                                        spacing='uniform',
                                        orientation='vertical')
        cb2.set_label(label)
        # Remove the outer bounds in tick labels
        ticks = [str(i) for i in bounds]
        ticks[0], ticks[-1] = '', ''
        cb2.ax.set_yticklabels(ticks)


    def text(self, text, beam, gate, fontsize=8):
        theta = (self.theta0 + beam * self.dtheta + 0.8 * self.dtheta) * np.pi / 180
        r = (self.r0 + gate * self.dr)
        plt.text(theta, r, text, fontsize=fontsize)


    def show(self):
        plt.tight_layout()
        plt.show()


    def save(self, filepath):
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()


    def _scale_plot(self, ax):
        # Scale min-max
        ax.set_thetamin(self.theta_ticks[0])
        ax.set_thetamax(self.theta_ticks[-1])
        ax.set_rmin(0)
        ax.set_rmax(self.r_ticks[-1])


    def _monotonically_increasing(self, vec):
        if len(vec) < 2:
            return True
        return all(x <= y for x, y in zip(vec[:-1], vec[1:]))


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
    # TODO 'diag' and 'spherical' may or may not work, haven't used them yet
    elif model.covariance_type == 'diag':
        covariances = np.diag(model.covariances_[n_cluster][[f1, f2]])
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
