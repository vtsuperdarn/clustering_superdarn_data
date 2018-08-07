"""
Grid-based DBSCAN
Author: Esther Robb

This is the fast implementation of Grid-based DBSCAN.
GBDBSCAN + Timefilter
If you don't want the timefilter, run it on 1 scan at a time.
"""
from algorithms.Algorithm import Algorithm
import numpy as np

UNCLASSIFIED = 0
NOISE = -1

class GBDBAlgorithm(Algorithm):

    def __init__(self, f, g, pts_ratio, ngate, nbeam, dr, dtheta, r_init=0):
        dtheta = dtheta * np.pi / 180.0
        self.C = np.zeros((ngate, nbeam))
        for gate in range(ngate):
            for beam in range(nbeam):
                # This is the ratio between radial and angular distance for each point. Across a row it's all the same, consider removing j.
                self.C[gate, beam] = self._calculate_ratio(dr, dtheta, gate, beam, r_init=r_init)
        self.g = g
        self.f = f
        print('Max beam_eps: ', np.max(self.g / (self.f * self.C)))
        print('Min beam_eps: ', np.min(self.g / (self.f * self.C)))
        self.pts_ratio = pts_ratio
        self.ngate = ngate
        self.nbeam = nbeam


    # Input for grid-based DBSCAN:
    # C matrix calculated based on sensor data.
    # There is very little variance from beam to beam for our radars - down to the 1e-16 level.
    def _calculate_ratio(self, dr, dt, i, j, r_init=0):
        r_init, dr, dt, i, j = float(r_init), float(dr), float(dt), float(i), float(j)
        cij = (r_init + dr * i) / (2.0 * dr) * (np.sin(dt * (j + 1.0) - dt * j) + np.sin(dt * j - dt * (j - 1.0)))
        return cij


    # add scan id here
    def _region_query(self, data, scan_i, grid_id):
        seeds = []
        hgt = self.g        #TODO should there be some rounding happening to accomidate discrete gate/wid?
        wid = self.g / (self.f * self.C[grid_id[0], grid_id[1]])
        ciel_hgt = int(np.ceil(hgt))
        ciel_wid = int(np.ceil(wid))

        # Check for neighbors in a box of shape ciel(2*wid), ciel(2*hgt) around the point
        g_min, g_max = max(0, grid_id[0] - ciel_hgt), min(self.ngate, grid_id[0] + ciel_hgt + 1)
        b_min, b_max = max(0, grid_id[1] - ciel_wid), min(self.nbeam, grid_id[1] + ciel_wid + 1)
        s_min, s_max = max(0, scan_i-1), min(len(data), scan_i+2)       # look at +- 1 scan
        possible_pts = 0
        for g in range(g_min, g_max):
            for b in range(b_min, b_max):
                new_id = (g, b)
                # Add the new point only if it falls within the ellipse defined by wid, hgt
                #if self._in_ellipse(new_id, grid_id, hgt, wid):
                for s in range(s_min, s_max):   # time filter
                    if self._in_ellipse(new_id, grid_id, hgt, wid):
                        possible_pts += 1
                        if data[s][new_id]:   # Add the point to seeds only if there is a 1 in the sparse matrix there
                            seeds.append((s, new_id))
        return seeds, possible_pts


    def _in_ellipse(self, p, q, hgt, wid):
        return ((q[0] - p[0])**2.0 / hgt**2.0 + (q[1] - p[1])**2.0 / wid**2.0) <= 1.0


    def _expand_cluster(self, data, grid_labels, scan_i, grid_id, cluster_id):
        seeds, possible_pts = self._region_query(data, scan_i, grid_id)

        k = possible_pts * self.pts_ratio
        if len(seeds) < k:
            grid_labels[scan_i][grid_id] = NOISE
            return False
        else:
            grid_labels[scan_i][grid_id] = cluster_id
            for seed_id in seeds:
                grid_labels[seed_id[0]][seed_id[1]] = cluster_id

            while len(seeds) > 0:
                current_scan, current_grid = seeds[0][0], seeds[0][1]
                results, possible_pts = self._region_query(data, current_scan, current_grid)
                k = possible_pts * self.pts_ratio
                if len(results) >= k:
                    for i in range(0, len(results)):
                        result_scan, result_point = results[i][0], results[i][1]
                        if grid_labels[result_scan][result_point] == UNCLASSIFIED or grid_labels[result_scan][result_point] == NOISE:
                            if grid_labels[result_scan][result_point] == UNCLASSIFIED:
                                seeds.append((result_scan, result_point))
                            grid_labels[result_scan][result_point] = cluster_id
                seeds = seeds[1:]
            return True


    def fit(self, data, data_i):
        """
        Inputs:
        m - A csr_sparse bool matrix, num_gates x num_beams x num_times
        m_i - indices where data can be found in the sparse matrix

        Outputs:
        An array with either a cluster id number or dbscan.NOISE (-1) for each
        column vector in m.
        """

        cluster_id = 1

        point_labels = []
        nscans = len(data)
        grid_labels = [np.zeros(data[0].shape).astype(int) for i in range(nscans)]

        # add another loop - for each scan
        for scan_i in range(nscans):
            m_i = data_i[scan_i]
            #scan_labels = np.zeros(data[scan_i].shape).astype(int)

            for grid_id in m_i:
                # Adaptively change one of the epsilon values and the min_points parameter using the C matrix
                if grid_labels[scan_i][grid_id] == UNCLASSIFIED:
                    if self._expand_cluster(data, grid_labels, scan_i, grid_id, cluster_id):
                        cluster_id = cluster_id + 1

            scan_pt_labels = [grid_labels[scan_i][grid_id] for grid_id in m_i]
            point_labels.append(scan_pt_labels)
        return point_labels



def dict_to_csr_sparse(data_dict, values):
    from scipy import sparse
    gate = data_dict['gate']
    beam = data_dict['beam']
    ngate = int(data_dict['nrang'])
    nbeam = int(data_dict['nbeam'])
    nscan = len(values)
    data = []
    data_i = []
    for i in range(nscan):
        m = sparse.csr_matrix((values[i], (gate[i], beam[i])), shape=(ngate, nbeam))
        m_i = list(zip(np.array(gate[i]).astype(int), np.array(beam[i]).astype(int)))
        data.append(m)
        data_i.append(m_i)
    return data, data_i


if __name__ == '__main__':
    """ Fake data 
    gate = np.random.randint(low=0, high=nrang-1, size=100)
    beam = np.random.randint(low=0, high=nbeam-1, size=100)
    data = np.row_stack((gate, beam))
    """
    """ Real radar data """
    import pickle
    rad_date = "sas_2018-02-07"
    ngate = 75
    nbeam = 16
    scan_i = 0            # ~1400 scans/day for SAS and CVW
    data_dict = pickle.load(open("../pickles/%s_scans.pickle" % rad_date, 'rb'))

    from scipy.stats import boxcox
    scans_to_use = range(100) #len(data_dict['gate']))
    values = [np.abs(data_dict['vel'][i]) for i in scans_to_use] #[[True]*len(data_dict['gate'][i]) for i in scans_to_use]
    data, data_i = dict_to_csr_sparse(data_dict, ngate, nbeam, values)


    """ Grid-based DBSCAN """
    from superdarn_cluster.FanPlot import FanPlot
    import matplotlib.pyplot as plt

    # Solid params board: f=0.2 0.3 0.4, g=1 2 3
    # Good way to tune this one: Use only the first 200 scans of SAS 2-7-18, because they tend to get all clustered together with bad params,
    # but then there is a long period of quiet so there's time separation between that big cluster and everything else.
    dr = 45
    dtheta = 3.3
    r_init = 180
    f = 0.2
    g = 1
    pts_ratio = 0.6
    eps2, d_eps = 30, 30
    gdb = GridBasedDBSCAN(f, g, eps2, d_eps, pts_ratio, ngate, nbeam, dr, dtheta, r_init)
    import time
    t0 = time.time()
    labels = gdb.fit(data, data_i)
    dt = time.time() - t0
    print('Time elapsed: %.2f s' % dt)

    unique_clusters = np.unique(np.hstack(labels))
    print('Grid-based DBSCAN Clusters: ', unique_clusters)
    cluster_colors = list(
        plt.cm.jet(np.linspace(0, 1, len(unique_clusters)+1)))  # one extra unused color at index 0 (no cluster label == 0)
    # randomly re-arrange colors for contrast in adjacent clusters
    np.random.seed(0)
    np.random.shuffle(cluster_colors)

    cluster_colors.append((0, 0, 0, 1))  # black for noise

    vel = data_dict['vel']
    for i in scans_to_use:
        clusters = np.unique(labels[i])
        # Plot a fanplot
        fanplot = FanPlot(nrange=ngate, nbeam=nbeam)
        for c in clusters:
            label_mask = labels[i] == c
            fanplot.plot(data_dict['beam'][i][label_mask], data_dict['gate'][i][label_mask], cluster_colors[c])
        plt.title('Grid-based DBSCAN fanplot\nf = %.2f    g = %d    pts_ratio = %.2f' % (f, g, pts_ratio))
        filename = '%s_f%.2f_g%d_ptRatio%.2f_scan%d_fanplot.png' % (rad_date, f, g, pts_ratio, i)
        # plt.show()
        plt.savefig(filename)
        plt.close()

        """ Velocity map """
        # Plot velocity fanplot
        fanplot = FanPlot(nrange=ngate, nbeam=nbeam)
        vel_step = 25
        vel_ranges = list(range(-200, 201, vel_step))
        vel_ranges.insert(0, -9999)
        vel_ranges.append(9999)
        cmap = plt.cm.jet       # use 'viridis' to make this redgreen colorblind proof
        vel_colors = cmap(np.linspace(0, 1, len(vel_ranges)))
        for s in range(len(vel_ranges) - 1):
            step_mask = (vel[i] >= vel_ranges[s]) & (vel[i] <= (vel_ranges[s+1]))
            fanplot.plot(data_dict['beam'][i][step_mask], data_dict['gate'][i][step_mask], vel_colors[s])

        filename = 'vel_scan%d_fanplot.png' % (i)
        fanplot.add_colorbar(vel_ranges, cmap)
        #plt.show()
        plt.savefig(filename)
        plt.close()



    """ Testing various params 
    pts_ratios = np.linspace(0.1, 1.0, 10)
    cij_min = _calculate_ratio(dr, dtheta * 3.1415926 / 180.0, i=0, j=0, r_init=r_init)
    cij_max = _calculate_ratio(dr, dtheta * 3.1415926 / 180.0, i=ngate-1, j=0, r_init=r_init)

    fs = np.linspace(0.1, 1, 9)#[1, 2, 3, 4.4] #np.linspace((1.0/cij_min), (1.0/cij_min)+1, 10)

    for f in fs:
        pts_ratio = 0.3
        #gs = range(int(np.ceil(5 * f)), int(np.ceil(nbeam * f)))        # 5 <= g/f <= num_beams
        gs = range(1, 11)    #
        for g in gs:
            gdb = GridBasedDBSCAN(float(f), float(g), pts_ratio, ngate, nbeam, dr, dtheta, r_init)
            # TODO why is this slow it should be fast, can I run it on a whole day?
            labels = gdb.fit(data['gate'], data['beam'])

            clusters = np.unique(labels)
            print('Grid-based DBSCAN Clusters: ', clusters)
            colors = list(plt.cm.plasma(np.linspace(0, 1, len(clusters))))  # one extra unused color at index 0 (no cluster label == 0)
            colors.append((0, 0, 0, 1)) # black for noise

            # Plot a fanplot
            fanplot = FanPlot(nrange=ngate, nbeam=nbeam)
            for c in clusters:
                label_mask = labels == c
                fanplot.plot(beam[label_mask], gate[label_mask], colors[c])
            plt.title('Grid-based DBSCAN fanplot\nf = %.2f    g = %d    pts_ratio = %.2f' % (f, g, pts_ratio))
            filename = '%s_f%.2f_g%d_ptRatio%.2f_fanplot.png' % (rad_date, f, g, pts_ratio)
            #plt.show()
            plt.savefig(filename)
            plt.close()
    """

    """ Regular DBSCAN 
    from sklearn.cluster import DBSCAN
    dbs_eps, min_pts = 5, 25
    dbs_data = np.column_stack((gate, beam))
    dbscan = DBSCAN(eps=dbs_eps, min_samples=min_pts)
    labels = dbscan.fit_predict(dbs_data)
    clusters = np.unique(labels)
    print('Regular DBSCAN Clusters: ', clusters)

    colors = list(plt.cm.plasma(np.linspace(0, 1, len(clusters))))  # one extra unused color at index 0 (no cluster label == 0)
    colors.append((0, 0, 0, 1)) # black for noise
    fanplot = FanPlot()
    for c in clusters:
        label_mask = labels == c
        fanplot.plot(beam[label_mask], gate[label_mask], colors[c])
    plt.title('Regular DBSCAN fanplot\neps = %.2f    k = %.2f' % (dbs_eps, min_pts))
    filename = '%s_eps%.2f_minPts%d_fanplot.png' % (rad_date, dbs_eps, min_pts)
    plt.savefig(filename)
    plt.close()
    for c in clusters:
        label_mask = labels == c
        plt.scatter(beam[label_mask], gate[label_mask], color=colors[c])
    plt.title('Regular DBSCAN gridplot (what regular DBSCAN sees)\neps = %.2f    k = %.2f' % (dbs_eps, min_pts))
    filename = '%s_eps%.2f_minPts%d_gridplot.png' % (rad_date, dbs_eps, min_pts)
    plt.savefig(filename)
    plt.close()

    """


