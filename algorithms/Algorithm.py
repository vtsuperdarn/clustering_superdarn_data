import pickle
import hashlib
import binascii
import datetime
from utilities.data_utils import get_data_dict_path
from utilities.classification_utils import *
import os

class Algorithm(object):
    """
    Superclass for algorithms.
    Contains data processing and plotting functions.
    """
    this_dir = os.path.abspath(os.path.dirname(__file__))

    # Public functions

    def __init__(self, start_time, end_time, rad, params, loadPickle=False):
        self.start_time = start_time
        self.end_time = end_time
        self.rad = rad
        self.params = params
        self.pickle_dir = self.this_dir + '/pickles/' + type(self).__name__
        if loadPickle:
            # Set instance vars equal to the pickled object's instance vars
            self.__dict__ = self._read_pickle()
        else:
            # Load data_dict from the data folder
            path = get_data_dict_path(start_time, rad)
            self.data_dict = pickle.load(open(path, 'rb'))


    def pickle(self):
        """
        Save a trained algorithm to ./pickles/<alg_name>/<parameter_hash>
        """
        # Create a directory unique to the subclass name
        if not os.path.exists(self.pickle_dir):
            os.makedirs(self.pickle_dir)
        picklefile = open(self._get_pickle_path(), 'wb')
        pickle.dump(self, picklefile)


    def plot_rti(self, beam, threshold, save=False):
        """
        Plot algorithm results on an RTI plot.
        Will plot color-coded clusters, IS/GS flags, and velocity heatmap.

        :param beam: an integer, or '*' to plot all beams
        :param threshold: which threshold to use.
                          'Ribiero', 'Blanchard code', or 'Blanchard paper'
        :param save: if False, show plot; if True, save plot to file
        :return:
        """
        # TODO add directory to save to
        import copy
        import matplotlib.pyplot as plt
        from utilities.plot_utils import plot_is_gs_colormesh, plot_clusters_colormesh, plot_vel_colormesh
        import matplotlib.dates as mdates

        gs_flg = np.hstack(self._classify(threshold))
        time_flat = np.hstack(np.array(self.data_dict['time']))
        unique_time = np.unique(time_flat)
        beams = np.hstack(np.array(self.data_dict['beam']))
        gates = np.hstack(np.array(self.data_dict['gate']))
        vels = np.hstack(np.array(self.data_dict['vel']))
        clust_flg = np.hstack(self.clust_flg)
        unique_clusters = np.unique(clust_flg)

        # TODO move this to plot_clusters_colormesh or do something else with it ... it assumes noise. need to standardize cluster #s
        # and not assumer there will always be nosie
        np.random.seed(0)
        clust_range = range(1, 4)

        alg = type(self).__name__
        date_str = self.start_time.strftime('%m-%d-%Y')
        ngate = self.data_dict['nrang']
        hours = mdates.HourLocator(byhour=range(0, 24, 4))

        fig = plt.figure(figsize=(14, 15))
        ax0 = plt.subplot(311)
        ax1 = plt.subplot(312)
        ax2 = plt.subplot(313)
        beam_filter = beam == beams

        plot_clusters_colormesh(ax0, unique_time, time_flat[beam_filter], gates[beam_filter], clust_range,
                                clust_flg[beam_filter], ngate)
        name = ('%s %s\t\t\t\tClusters\t\t\t\t%s\t\t\t\tbeam %d' \
               % (self.rad.upper(), date_str, type(self).__name__, beam)).expandtabs()
        ax0.set_title(name)
        ax0.xaxis.set_major_locator(hours)
        plot_is_gs_colormesh(ax1, unique_time, time_flat[beam_filter], gates[beam_filter], gs_flg[beam_filter],
                             ngate,
                             plot_indeterminate=False, plot_closerange=True)
        name = ('%s %s\t\t\t\tIS/GS\t\t\t\t%s / %s threshold\t\t\t\tbeam %d' \
               % (self.rad.upper(), date_str, alg, threshold, beam)).expandtabs()
        ax1.set_title(name)
        ax1.xaxis.set_major_locator(hours)
        plot_vel_colormesh(fig, ax2, unique_time, time_flat[beam_filter], gates[beam_filter], vels[beam_filter],
                           ngate)
        name = ('%s %s\t\t\t\t\t\t\t\tVelocity\t\t\t\t\t\t\t\tbeam %d' \
               % (self.rad.upper(), date_str, beam)).expandtabs()
        ax2.set_title(name)
        ax2.xaxis.set_major_locator(hours)
        plt.show()
        #plt.savefig('%s/%s_%d%02d%02d_%02d.jpg' % (rti_dir, rad, yr, mo, day, b))
        #fig.clf()  # Necessary to prevent memory explosion

    def plot_fanplot(self, start_time, end_time):
        # TODO
        return

    # Private functions

    def _classify(self, threshold):
        """
        Classify each cluster created by the algorithm
        :param threshold: 'Ribiero' or 'Blanchard code' or 'Blanchard paper'
        :return: GS labels for each scatter point
        """
        # Use abs value of vel & width
        vel = np.hstack(np.abs(self.data_dict['vel']))
        t = np.hstack(self.data_dict['time'])
        wid = np.hstack(np.abs(self.data_dict['wid']))
        clust_flg_1d = np.hstack(self.clust_flg)
        gs_flg = np.zeros(len(clust_flg_1d))
        # Classify each cluster
        for c in np.unique(clust_flg_1d):
            clust_mask = c == clust_flg_1d
            if c == -1:
                gs_flg[clust_mask] = -1  # Noise flag
            else:
                if threshold == 'Blanchard code':
                    gs_flg[clust_mask] = blanchard_gs_flg(vel[clust_mask], wid[clust_mask], 'code')
                elif threshold == 'Blanchard paper':
                    gs_flg[clust_mask] = blanchard_gs_flg(vel[clust_mask], wid[clust_mask], 'paper')
                elif threshold == 'Ribiero':
                    gs_flg[clust_mask] = ribiero_gs_flg(vel[clust_mask], t[clust_mask])
        # Convert 1D array to list of scans and return
        return self._1D_to_scanxscan(gs_flg)


    def _1D_to_scanxscan(self, array):
        """
        Convert a 1-dimensional array to a list of data from each scan
        :param array: a 1D array, length = <total # points in data_dict>
        :return: a list of arrays, shape: <number of scans> x <data points for each scan>
        """
        scans = []
        i = 0
        for s in self.data_dict['gate']:
            scans.append(array[i:i+len(s)])
            i += len(s)
        return scans


    def _get_pickle_path(self):
        """
        Get path to the unique pickle file for an object with this time/radar/params/algorithm
        :return: path to pickle file (string)
        """
        # Create a unique filename based on params
        m = hashlib.md5()
        m.update(str(self.start_time).encode('ascii'))
        m.update(str(self.end_time).encode('ascii'))
        m.update(self.rad.encode('ascii'))
        m.update(str(self.params).encode('ascii'))  # algorithm parameters
        hash_str = binascii.hexlify(m.digest()).decode('ascii')
        # Save the pickle
        return self.pickle_dir + '/' + hash_str + '.pickle'


    def _read_pickle(self):
        try:
            picklefile = open(self._get_pickle_path(), 'rb')
            new_obj = pickle.load(picklefile)
            return new_obj.__dict__    # Instance vars of the pickled object
        except FileNotFoundError:
            raise Exception('No pickle file found for this time/radar/params/algorithm')




