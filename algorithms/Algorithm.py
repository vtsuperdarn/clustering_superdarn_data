import pickle
from utilities.data_utils import get_data_dict_path
from utilities.plot_utils import RangeTimePlot, FanPlot
from utilities.classification_utils import *
import os
from matplotlib.dates import date2num
import datetime
import copy


class Algorithm(object):
    """
    Superclass for algorithms.
    Contains data processing and plotting functions.
    """
    this_dir = os.path.abspath(os.path.dirname(__file__))
    base_path_dir = this_dir + '/../plots'

    # Public functions

    def __init__(self, start_time, end_time, rad, params, useSavedResult=False):
        self.start_time = start_time
        self.end_time = end_time
        self.rad = rad
        self.params = params
        self.pickle_dir = self.this_dir + '/pickles/' + type(self).__name__
        if useSavedResult:
            # Set instance vars equal to the pickled object's instance vars
            self.__dict__ = self._read_pickle()
        else:
            # Load data_dict from the data folder
            path = get_data_dict_path(start_time, rad)
            self.data_dict = pickle.load(open(path, 'rb'))


    def save_result(self):
        """
        Save a trained algorithm to ./pickles/<alg_name>/<parameter_hash>
        """
        # Create a directory unique to the subclass name
        if not os.path.exists(self.pickle_dir):
            os.makedirs(self.pickle_dir)
        picklefile = open(self._get_pickle_path(), 'wb')
        pickle.dump(self, picklefile)


    def plot_rti(self, beam, threshold, vel_max=200, vel_step=25, show=True, save=False):
        unique_times = np.unique(np.hstack(self.data_dict['time']))
        nrang = self.data_dict['nrang']
        gs_flg = np.hstack(self._classify(threshold))
        # Set up plot names
        date_str = self.start_time.strftime('%m-%d-%Y')
        alg = type(self).__name__
        clust_name = ('%s %s\t\t\t\t%d clusters\t\t\t\t%s\t\t\t\tbeam %d'
                        % (self.rad.upper(), date_str,
                           len(np.unique(np.hstack(self.clust_flg))),
                           alg, beam)
                      ).expandtabs()
        isgs_name = ('%s %s\t\t\t\tIS/GS\t\t\t\t%s / %s threshold\t\t\t\tbeam %d'
                        % (self.rad.upper(), date_str, alg, threshold, beam)
                     ).expandtabs()
        vel_name = ('%s %s\t\t\t\t\t\t\t\tVelocity\t\t\t\t\t\t\t\tbeam %d'
                        % (self.rad.upper(), date_str, beam)
                    ).expandtabs()
        # Create and show subplots
        rtp = RangeTimePlot(nrang, unique_times)
        rtp.addClusterPlot(self.data_dict, self.clust_flg, beam, clust_name)
        rtp.addGSISPlot(self.data_dict, gs_flg, beam, isgs_name)
        rtp.addVelPlot(self.data_dict, beam, vel_name, vel_max=vel_max, vel_step=vel_step)
        if save:
            plot_date = self.start_time.strftime('%Y%m%d')
            filename = '%s_%s_%s_%s.jpg' % (self.rad, plot_date, beam,
                                            threshold.replace(' ', '').lower())
            filepath = self._get_plot_path(alg, 'rti') + '/' + filename
            rtp.save(filepath)
        if show:
            rtp.show()
        rtp.close()


    def plot_fanplots(self, start_time, end_time, vel_max=200, vel_step=25, show=True, save=False):
        # Find the corresponding scans
        s, e = date2num(start_time), date2num(end_time)
        scan_start = None
        scan_end = None
        for i, t in enumerate(self.data_dict['time']):
            if (np.sum(s <= t) > 0) and not scan_start:     # reached start point
                scan_start = i
            if (np.sum(e <= t) > 0) and scan_start:         # reached end point
                scan_end = i
                break
        if (scan_start == None) or (scan_end == None):
            raise Exception('%s thru %s is not contained in data' % (start_time, end_time))
        # Create the fanplots

        date_str = self.start_time.strftime('%m-%d-%Y')
        alg = type(self).__name__

        if save:       # Only create the directory path if savePlot is true
            plot_date = self.start_time.strftime('%Y%m%d')
            filename = '%s_%s' % (self.rad, plot_date)      # Scan time and .jpg will be added to this by plot_clusters
            base_filepath = self._get_plot_path(alg, 'fanplot') + '/' + filename
        else:
            base_filepath = None

        fan_name = ('%s %s\t\t%d clusters\t\t%s\t\t'
                      % (self.rad.upper(), date_str,
                         len(np.unique(np.hstack(self.clust_flg))),
                         alg)
                      ).expandtabs()
        fanplot = FanPlot(self.data_dict['nrang'], self.data_dict['nbeam'])
        fanplot.plot_clusters(self.data_dict, self.clust_flg,
                              range(scan_start, scan_end+1),
                              vel_max=vel_max, vel_step=vel_step,
                              name=fan_name, show=show, save=save,
                              base_filepath=base_filepath)


    # Private functions

    def _get_plot_path(self, alg, plot_type):
        today = datetime.datetime.now().strftime('%m-%d-%Y')
        dir = '%s/%s %s/%s' \
                  % (self.base_path_dir, today, alg, plot_type)
        if not os.path.exists(dir):
            os.makedirs(dir)
        return dir


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
                else:
                    raise Exception('Bad threshold %s' % str(threshold))
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


    # TODO add a function to print loadable params
    def _get_pickle_path(self):
        """
        Get path to the unique pickle file for an object with this time/radar/params/algorithm
        :return: path to pickle file (string)
        """
        # Create a unique filename based on params
        filename = '%s_%s_%s_%s' % (self.rad,
                                    self.start_time.strftime('%Y%m%d-%H:%M:%S'),
                                    self.end_time.strftime('%Y%m%d-%H:%M:%S'),
                                    str(self.params))
        # Save the pickle
        return self.pickle_dir + '/' + filename + '.pickle'


    def _read_pickle(self):
        try:
            picklefile = open(self._get_pickle_path(), 'rb')
            new_obj = pickle.load(picklefile)
            return new_obj.__dict__    # Instance vars of the pickled object
        except FileNotFoundError:
            raise Exception('No pickle file found for this time/radar/params/algorithm')


    def _randomize_flags(self, flags):
        """
        Randomize flags so that plot colors are randomized
        DBSCAN can create >1000 clusters, and physically clusters are usually similar in cluster number,
        so they will be plotted in a similar shade which may not be distinguishable from their neighbors.
        Randomizing the cluster numbers will reduce (but not eliminate) this problem.

        :param flags: 1D array of flags
        :return: 1D array of randomly re-assigned flags (cluster shapes are the same, flag #s have been changed)
        """
        # Randomize flags so that plot colors are randomized
        unique_flgs = np.unique(flags)
        clust_range = list(range(int(min(unique_flgs)), int(max(unique_flgs)) + 1))
        flg_randomizer = copy.deepcopy(clust_range)
        np.random.seed(0)   # Subsequent runs will produce the same flags
        np.random.shuffle(flg_randomizer)
        random_flags = np.zeros(len(flags))
        for f in unique_flgs:
            cluster_mask = f == flags
            if f == -1:
                random_flags[cluster_mask] = -1
            else:
                random_flags[cluster_mask] = flg_randomizer[f]
        return random_flags


from sklearn.mixture import GaussianMixture
from scipy.stats import boxcox
import time

class GMMAlgorithm(Algorithm):
    """
    Superclass holding shared functions for algorithms that use GMM at some point.
    """
    def __init__(self, start_time, end_time, rad, params, useSavedResult):
        super().__init__(start_time, end_time, rad, params, useSavedResult=useSavedResult)


    def _gmm(self, data):
        n_clusters = self.params['n_clusters']
        cov = self.params['cov']
        estimator = GaussianMixture(n_components=n_clusters,
                                    covariance_type=cov, max_iter=500,
                                    random_state=0, n_init=5, init_params='kmeans')
        t0 = time.time()
        estimator.fit(data)
        runtime = time.time() - t0
        clust_flg = estimator.predict(data)
        return clust_flg, runtime


    def _get_gmm_data_array(self):
        data = []
        for feature in self.params['features']:
            vals = self.data_dict[feature]
            if self.params['BoxCox'] and (feature == 'vel' or feature == 'wid'):
                vals = boxcox(vals)[0]
            data.append(np.hstack(vals))
        return np.column_stack(data)



