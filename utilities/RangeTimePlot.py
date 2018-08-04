import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter


# TODO move to plot_utils

class RangeTimePlot(object):
    def __init__(self, nrang, unique_times, num_subplots=3):
        self.nrang = nrang
        self.unique_gates = np.linspace(1, nrang, nrang)
        self.unique_times = unique_times
        self.num_subplots = num_subplots
        self.axes = []
        self.fig = plt.figure(num=99, figsize=(14, 5*num_subplots))


    def addClusterPlot(self, data_dict, clust_flg, beam, show_closerange=True):
        # add new axis
        ax = self._add_axis()
        # set up variables for plotter
        time = np.hstack(data_dict['time'])
        gate = np.hstack(data_dict['gate'])
        allbeam = np.hstack(data_dict['beam'])
        flags = np.hstack(clust_flg)
        mask = allbeam == beam
        if show_closerange:
            mask = mask & (gate > 10)
        if -1 in flags:                     # contains noise flag
            cmap = plt.cm.nipy_spectral     # includes white and black at edges
            n_clusters = np.max(flags)
            bounds = list(range(n_clusters+2))
            bounds_noise_adjust = 10 + int(n_clusters / 10) # TODO what should this be?
            bounds.append(bounds[-1] + bounds_noise_adjust)                 # white
            bounds.insert((bounds[0] - bounds_noise_adjust), 0)             # black
            flags[flags == -1] = (bounds[0] - bounds_noise_adjust)          # set noise black
        else:
            cmap = plt.cm.hsv                       # no black or white
            n_clusters = np.max(clust_flg)
            bounds = list(range(n_clusters+1))       # add 1 in case labels start at 0
        self._create_colormesh(ax, time, gate, flags, mask, bounds, cmap)
        return ax

    def addGSISPlot(self, data_dict, gs_flg, beam, show_closerange=True):
        # add new axis
        ax = self._add_axis()
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
            bounds = [-1, 0, 1, 2] # TODO may need some outlier bounds?
        else:
            cmap = mpl.colors.ListedColormap([(0.0, 0.0, 1.0, 1.0),  # blue
                                       (1.0, 0.0, 0.0, 1.0)])  # red
            bounds = [0, 1, 2]  # TODO may need some outlier bounds?
        self._create_colormesh(ax, time, gate, flags, mask, bounds, cmap)
        return ax


    def addVelPlot(self):
        pass


    def show(self):
        plt.figure(num=99)
        plt.show()

    def _add_axis(self):
        plt.figure(num=99)
        ax_i = len(self.axes) + 1
        ax = plt.subplot(self.num_subplots, 1, ax_i)
        self.axes.append(ax)
        return ax

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