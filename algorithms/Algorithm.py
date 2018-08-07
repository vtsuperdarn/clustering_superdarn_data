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
            data_dict = pickle.load(open(path, 'rb'))
            self.data_dict = self._filter_by_time(start_time, end_time, data_dict)


    def save_result(self):
        """
        Save a trained algorithm to ./pickles/<alg_name>/<parameter_hash>
        """
        # Create a directory unique to the subclass name
        if not os.path.exists(self.pickle_dir):
            os.makedirs(self.pickle_dir)
        picklefile = open(self._get_pickle_path(), 'wb')
        pickle.dump(self, picklefile)


    def plot_rti_traditional(self, beam, vel_max=200, vel_step=25, show=True, save=False):
        unique_times = np.unique(np.hstack(self.data_dict['time']))
        nrang = self.data_dict['nrang']
        gs_flg = np.hstack(self.data_dict['trad_gsflg'])
        alg = type(self).__name__
        # Set up plot names
        date_str = self.start_time.strftime('%m-%d-%Y')
        isgs_name = ('%s %s\t\t\t\tIS/GS\t\t\t\tTraditional classification\t\t\t\tbeam %d'
                     % (self.rad.upper(), date_str, beam)
                     ).expandtabs()
        vel_name = ('%s %s\t\t\t\t\t\t\t\tVelocity\t\t\t\t\t\t\t\tbeam %d'
                    % (self.rad.upper(), date_str, beam)
                    ).expandtabs()
        # Create and show subplots
        rtp = RangeTimePlot(nrang, unique_times)
        rtp.addGSISPlot(self.data_dict, gs_flg, beam, isgs_name)
        rtp.addVelPlot(self.data_dict, beam, vel_name, vel_max=vel_max, vel_step=vel_step)
        if save:
            plot_date = self.start_time.strftime('%Y%m%d')
            filename = '%s_%s_%s_traditional.jpg' % (self.rad, plot_date, beam)
            filepath = self._get_plot_path(alg, 'rti') + '/' + filename
            rtp.save(filepath)
        if show:
            rtp.show()
        rtp.close()


    def plot_rti(self, beam, threshold, vel_max=200, vel_step=25, show=True, save=False):
        unique_times = np.unique(np.hstack(self.data_dict['time']))
        nrang = self.data_dict['nrang']
        gs_flg = np.hstack(self._classify(threshold))
        # Set up plot names
        date_str = self.start_time.strftime('%m-%d-%Y')
        alg = type(self).__name__

        if beam == '*':
            beams = range(int(self.data_dict['nbeam']))
        else:
            beams = [beam]
        for b in beams:
            clust_name = ('%s %s\t\t\t\t%d clusters\t\t\t\t%s\t\t\t\tbeam %d'
                            % (self.rad.upper(), date_str,
                               len(np.unique(np.hstack(self.clust_flg))),
                               alg, b)
                          ).expandtabs()
            isgs_name = ('%s %s\t\t\t\tIS/GS\t\t\t\t%s / %s threshold\t\t\t\tbeam %d'
                            % (self.rad.upper(), date_str, alg, threshold, b)
                         ).expandtabs()
            vel_name = ('%s %s\t\t\t\t\t\t\t\tVelocity\t\t\t\t\t\t\t\tbeam %d'
                            % (self.rad.upper(), date_str, b)
                        ).expandtabs()
            # Create and show subplots
            rtp = RangeTimePlot(nrang, unique_times)
            rtp.addClusterPlot(self.data_dict, self.clust_flg, b, clust_name)
            rtp.addGSISPlot(self.data_dict, gs_flg, b, isgs_name)
            rtp.addVelPlot(self.data_dict, b, vel_name, vel_max=vel_max, vel_step=vel_step)
            if save:
                plot_date = self.start_time.strftime('%Y%m%d')
                filename = '%s_%s_%s_%s.jpg' % (self.rad, plot_date, b,
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

    def _filter_by_time(self, start_time, end_time, data_dict):
        time = data_dict['time']
        start_i, end_i = None, None
        start_time, end_time = date2num(start_time), date2num(end_time)
        if start_time < time[0][0]: # Sometimes start time is a few seconds before the first scan
            start_time = time[0][0]
        for i, t in enumerate(time):
            if np.sum(start_time >= t) > 0 and start_i == None:
                start_i = i
            if np.sum(end_time >= t) > 0 and start_i != None:
                end_i = i+1
        data_dict['gate'] = data_dict['gate'][start_i:end_i]
        data_dict['time'] = data_dict['time'][start_i:end_i]
        data_dict['beam'] = data_dict['beam'][start_i:end_i]
        data_dict['vel'] = data_dict['vel'][start_i:end_i]
        data_dict['wid'] = data_dict['wid'][start_i:end_i]
        data_dict['elv'] = data_dict['elv'][start_i:end_i]
        data_dict['trad_gsflg'] = data_dict['trad_gsflg'][start_i:end_i]
        return data_dict


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


    def _get_pickle_path(self):
        """
        Get path to the unique pickle file for an object with this time/radar/params/algorithm
        :return: path to pickle file (string)
        """
        # Create a unique filename based on params/date/radar
        params = '{'
        for i, key in enumerate(sorted(self.params.keys())):          # Sort so that the order is not random
            params += "%s: %s" % (key, self.params[key])
            if i != len(self.params.keys()) - 1:
                params += ', '
        params += '}'
        filename = '%s_%s_%s_%s' % (self.rad,
                                    self.start_time.strftime('%Y%m%d-%H:%M:%S'),
                                    self.end_time.strftime('%Y%m%d-%H:%M:%S'),
                                    params)
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


class Traditional(Algorithm):
    """
    Stub class used just to plot the traditional method.
    Initialize it and call plot_traditional_rti (also available from any other algorithm class)
    """
    def __init__(self, start_time, end_time, rad):
        super().__init__(start_time, end_time, rad, params={}, useSavedResult=False)



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


    def _gmm_on_existing_clusters(self, data, clust_flg):
        labels_unique = np.unique(clust_flg)
        runtime = 0
        for c in labels_unique:
            gb_cluster_mask = (clust_flg == c)
            num_pts = np.sum(gb_cluster_mask)
            if num_pts < 500 or c == -1:
                continue
            gmm_labels, gmm_runtime = self._gmm(data[gb_cluster_mask])
            runtime += gmm_runtime
            gmm_labels += np.max(clust_flg) + 1
            clust_flg[gb_cluster_mask] = gmm_labels
        return clust_flg, runtime


    def _get_gmm_data_array(self):
        data = []
        for feature in self.params['features']:
            vals = self.data_dict[feature]
            if self.params['BoxCox'] and (feature == 'vel' or feature == 'wid'):
                vals = boxcox(vals)[0]
            data.append(np.hstack(vals))
        return np.column_stack(data)





class GBDBAlgorithm(Algorithm):
    """
    Grid-based DBSCAN
    Based on Kellner et al. 2012

    This is the fast implementation of Grid-based DBSCAN, with a timefilter added in.
    If you don't want the timefilter, run it on just 1 scan.
    """

    """
    GBDB params dict keys:
    f, g, pts_ratio

    Other class variables unique to GBDB:
    self.C
    self.r_init
    self.dr
    self.dtheta
    """

    # TODO make this so that it labels the first cluster 0, like other algorithms in this library
    # TODO remove the damn sparse matrix, it's useless?
    UNCLASSIFIED = 0
    NOISE = -1

    def __init__(self, start_time, end_time, rad, params, useSavedResult):
        # Call superclass constructor to get data_dict and save params
        super().__init__(start_time, end_time, rad,
                         params, useSavedResult=useSavedResult)
        if not useSavedResult:
            # Create the C matrix - ratio of radial / angular distance for each point
            dtheta = self.params['dtheta'] * np.pi / 180.0
            nrang, nbeam = int(self.data_dict['nrang']), int(self.data_dict['nbeam'])
            self.C = np.zeros((nrang, nbeam))
            for gate in range(nrang):
                for beam in range(nbeam):
                    self.C[gate, beam] = self._calculate_ratio(self.params['dr'], dtheta, gate, beam,
                                                               r_init=self.params['r_init'])

    def _gbdb(self, data, data_i):
        t0 = time.time()
        cluster_id = 1
        clust_flgs = []
        nscans = len(data)
        grid_labels = [np.zeros(data[0].shape).astype(int) for i in range(nscans)]
        for scan_i in range(nscans):
            m_i = data_i[scan_i]
            for grid_id in m_i:
                if grid_labels[scan_i][grid_id] == self.UNCLASSIFIED:
                    if self._expand_cluster(data, grid_labels, scan_i, grid_id, cluster_id):
                        cluster_id = cluster_id + 1
            scan_pt_labels = [grid_labels[scan_i][grid_id] for grid_id in m_i]
            clust_flgs.extend(scan_pt_labels)
        runtime = time.time() - t0
        return np.array(clust_flgs), runtime


    def _get_gbdb_data_matrix(self, data_dict):
        from scipy import sparse
        gate = data_dict['gate']
        beam = data_dict['beam']
        values = [[True]*len(s) for s in beam]
        ngate = int(data_dict['nrang'])
        nbeam = int(data_dict['nbeam'])
        nscan = len(beam)
        data = []
        data_i = []
        for i in range(nscan):
            m = sparse.csr_matrix((values[i], (gate[i], beam[i])), shape=(ngate, nbeam))
            m_i = list(zip(np.array(gate[i]).astype(int), np.array(beam[i]).astype(int)))
            data.append(m)
            data_i.append(m_i)
        return data, data_i


    def _expand_cluster(self, data, grid_labels, scan_i, grid_id, cluster_id):
        seeds, possible_pts = self._region_query(data, scan_i, grid_id)
        k = possible_pts * self.params['pts_ratio']
        if len(seeds) < k:
            grid_labels[scan_i][grid_id] = self.NOISE
            return False
        else:
            grid_labels[scan_i][grid_id] = cluster_id
            for seed_id in seeds:
                grid_labels[seed_id[0]][seed_id[1]] = cluster_id

            while len(seeds) > 0:
                current_scan, current_grid = seeds[0][0], seeds[0][1]
                results, possible_pts = self._region_query(data, current_scan, current_grid)
                k = possible_pts * self.params['pts_ratio']
                if len(results) >= k:
                    for i in range(0, len(results)):
                        result_scan, result_point = results[i][0], results[i][1]
                        if grid_labels[result_scan][result_point] == self.UNCLASSIFIED \
                                or grid_labels[result_scan][result_point] == self.NOISE:
                            if grid_labels[result_scan][result_point] == self.UNCLASSIFIED:
                                seeds.append((result_scan, result_point))
                            grid_labels[result_scan][result_point] = cluster_id
                seeds = seeds[1:]
            return True


    def _region_query(self, data, scan_i, grid_id):
        seeds = []
        hgt = self.params['g']
        wid = self.params['g'] / (self.params['f'] * self.C[grid_id[0], grid_id[1]])
        ciel_hgt = int(np.ceil(hgt))
        ciel_wid = int(np.ceil(wid))

        # Check for neighbors in a box of shape ciel(2*wid), ciel(2*hgt) around the point
        g_min = max(0, grid_id[0] - ciel_hgt)     # gate box
        g_max = min(int(self.data_dict['nrang']), grid_id[0] + ciel_hgt + 1)
        b_min = max(0, grid_id[1] - ciel_wid)     # beam box
        b_max = min(int(self.data_dict['nbeam']), grid_id[1] + ciel_wid + 1)
        s_min = max(0, scan_i - self.params['scan_eps'])  # scan box
        s_max = min(len(data), scan_i + self.params['scan_eps']+1)
        possible_pts = 0
        for g in range(g_min, g_max):
            for b in range(b_min, b_max):
                new_id = (g, b)
                # Add the new point only if it falls within the ellipse defined by wid, hgt
                for s in range(s_min, s_max):   # time filter
                    if self._in_ellipse(new_id, grid_id, hgt, wid):
                        possible_pts += 1
                        if data[s][new_id]:   # Add the point to seeds only if there is a 1 in the sparse matrix there
                            seeds.append((s, new_id))
        return seeds, possible_pts


    def _in_ellipse(self, p, q, hgt, wid):
        return ((q[0] - p[0])**2.0 / hgt**2.0 + (q[1] - p[1])**2.0 / wid**2.0) <= 1.0


    # This is the ratio between radial and angular distance for some point on the grid.
    # There is very little variance from beam to beam for our radars - down to the 1e-16 level.
    # So the rows of this have minimal effect.
    def _calculate_ratio(self, dr, dt, i, j, r_init=0):
        r_init, dr, dt, i, j = float(r_init), float(dr), float(dt), float(i), float(j)
        cij = (r_init + dr * i) / (2.0 * dr) * (np.sin(dt * (j + 1.0) - dt * j) + np.sin(dt * j - dt * (j - 1.0)))
        return cij




