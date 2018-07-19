"""
Grid-based DBSCAN
Author: Esther Robb

This is the fast implementation of Grid-based DBSCAN.
It uses a sparse Boolean array of data of size (num_grids) x (num_beams)
The data structure is why it is able to run faster - instead of checking all points to
find neighbors, it only checks adjacent points.

Complete implementation.
Confirmed to give the same output as GridBasedDBSCAN_simple.py.
"""

import numpy as np

UNCLASSIFIED = 0
NOISE = -1


class GridBasedDBSCAN():

    def __init__(self, f, g, eps2, d_eps, pts_ratio, ngate, nbeam, dr, dtheta, r_init=0):
        dtheta = dtheta * np.pi / 180.0
        self.C = np.zeros((ngate, nbeam))
        for gate in range(ngate):
            for beam in range(nbeam):
                # This is the ratio between radial and angular distance for each point. Across a row it's all the same, consider removing j.
                self.C[gate, beam] = self._calculate_ratio(dr, dtheta, gate, beam, r_init=r_init)
        self.g = g
        self.f = f
        self.eps2 = eps2
        self.d_eps = d_eps
        print('Max beam_eps (f=%.2f, g=%d): %2.f' % (f, g, np.max(self.g / (self.f * self.C))))
        print('Min beam_eps: (f=%.2f, g=%d): %.2f' % (f, g, np.min(self.g / (self.f * self.C))))
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


    def _region_query(self, m, grid_id):
        seeds = []
        hgt = self.g        #TODO should there be some rounding happening to accomidate discrete gate/wid?
        wid = self.g / (self.f * self.C[grid_id[0], grid_id[1]])
        ciel_hgt = int(np.ceil(hgt))
        ciel_wid = int(np.ceil(wid))

        # Check for neighbors in a box of shape ciel(2*wid), ciel(2*hgt) around the point
        g_min, g_max = max(0, grid_id[0] - ciel_hgt), min(self.ngate, grid_id[0] + ciel_hgt + 1)
        b_min, b_max = max(0, grid_id[1] - ciel_wid), min(self.nbeam, grid_id[1] + ciel_wid + 1)
        possible_pts = 0
        for g in range(g_min, g_max):
            for b in range(b_min, b_max):
                new_id = (g, b)
                # Add the new point only if it falls within the ellipse defined by wid, hgt
                if self._in_ellipse(new_id, grid_id, hgt, wid):
                    possible_pts += 1
                    if m[new_id]:   # Add the point to seeds only if there is a 1 in the sparse matrix there
                        if np.abs(m[grid_id] - m[new_id]) <= self.eps2:
                            seeds.append(new_id)
        return seeds, possible_pts


    def _in_ellipse(self, p, q, hgt, wid):
        return ((q[0] - p[0])**2.0 / hgt**2.0 + (q[1] - p[1])**2.0 / wid**2.0) <= 1.0


    def _cluster_avg(self, data, grid_labels, cluster_id):
        cluster_mask = grid_labels == cluster_id
        data_i = data[cluster_mask]
        sum = data_i.sum()
        size = data_i.shape[1]
        return sum/size, size


    def _expand_cluster(self, m, grid_labels, grid_id, cluster_id):
        seeds, possible_pts = self._region_query(m, grid_id)
        k = possible_pts * self.pts_ratio
        if len(seeds) < k:
            grid_labels[grid_id] = NOISE
            return False
        else:
            grid_labels[grid_id] = cluster_id
            for seed_id in seeds:
                grid_labels[seed_id] = cluster_id

            cluster_avg, cluster_size = self._cluster_avg(m, grid_labels, cluster_id)

            while len(seeds) > 0:
                current_point = seeds[0]
                results, possible_pts = self._region_query(m, current_point)
                k = possible_pts * self.pts_ratio
                if len(results) >= k:
                    for i in range(0, len(results)):
                        result_point = results[i]
                        if grid_labels[result_point] == UNCLASSIFIED or grid_labels[result_point] == NOISE:
                            # Check the new point against the cluster avg using d_eps
                            if np.abs(cluster_avg - m[result_point]) <= self.d_eps:
                                # If this point has not been visited before (not previously classified as noise), you should
                                # add it to seeds to find all its neighbors.
                                if grid_labels[result_point] == UNCLASSIFIED:
                                    seeds.append(result_point)
                                grid_labels[result_point] = cluster_id
                                # Update the cluster size and average (without looping through the whole dataset again)
                                cluster_avg = (cluster_avg * cluster_size + m[result_point]) / (cluster_size + 1)
                                cluster_size += 1
                seeds = seeds[1:]
            return True


    def fit(self, m, m_i):
        """
        Inputs:
        m - A csr_sparse bool matrix, num_gates x num_beams x num_times
        m_i - indices where data can be found in the sparse matrix

        Outputs:
        An array with either a cluster id number or dbscan.NOISE (-1) for each
        column vector in m.
        """
        from scipy import sparse

        #m = sparse.csr_matrix((np.array([True] * len(gate)), (gate, beam)), shape=(self.ngate, self.nbeam))
        #m_i = list(zip(gate, beam))
        grid_labels = np.zeros(m.shape).astype(int)  # TODO sparsify
        grid_labels[:, :] = UNCLASSIFIED

        cluster_id = 1

        for grid_id in m_i:
            # Adaptively change one of the epsilon values and the min_points parameter using the C matrix
            if grid_labels[grid_id] == UNCLASSIFIED:
                if self._expand_cluster(m, grid_labels, grid_id, cluster_id):
                    cluster_id = cluster_id + 1

        point_labels = [grid_labels[grid_id] for grid_id in m_i]
        return point_labels


def _calculate_ratio(dr, dt, i, j, r_init=0):
    r_init, dr, dt, i, j = float(r_init), float(dr), float(dt), float(i), float(j)
    cij = (r_init + dr * i) / (2.0 * dr) * (np.sin(dt * (j + 1.0) - dt * j) + np.sin(dt * j - dt * (j - 1.0)))
    return cij

# TODO add a 'values' param instead of using something from the dictionary
def dict_to_csr_sparse(data_dict, ngate, nbeam, values):
    from scipy import sparse
    gate = data_dict['gate']
    beam = data_dict['beam']
    nscan = len(values)
    data = []
    data_i = []
    for i in range(nscan):
        m = sparse.csr_matrix((values[i], (gate[i], beam[i])), shape=(ngate, nbeam))
        m_i = list(zip(np.array(gate[i]).astype(int), np.array(beam[i]).astype(int)))
        data.append(m)
        data_i.append(m_i)
    return data, data_i

# TODO what about when there's no data... this assumes there is always data
# TODO I don't think the non-spatial value part of this is working well, just based on the velocity plot
# TODO is the time part of this working properly? Looks to be. Try a time epsilon? Smaller g?

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
    # TODO turn this into a sparse matrix? I can't find a good way to *find* the data in a sparse matrix but there must be a way...
    data_dict = pickle.load(open("../pickles/%s_scans.pickle" % rad_date, 'rb'))

    #from scipy.stats import boxcox

    scans_to_use = range(10) #range(len(data_dict['vel']))
    values = [np.abs(v) for v in data_dict['vel']] #[[True] * len(data_dict['gate'][i]) for i in scans_to_use]
    data, data_i = dict_to_csr_sparse(data_dict, ngate, nbeam, values)

    """ Grid-based DBSCAN """
    from superdarn_cluster.FanPlot import FanPlot
    import matplotlib.pyplot as plt

    # Solid params across the board: f=0.3, g=2
    dr = 45
    dtheta = 3.3
    r_init = 180
    f = 0.3
    g = 2
    eps2 = 5
    d_eps = 10
    pts_ratio = 0.3
    gdb = GridBasedDBSCAN(f, g, eps2, d_eps, pts_ratio, ngate, nbeam, dr, dtheta, r_init)

    import time
    t = 0
    vel = data_dict['vel']
    for i in scans_to_use:

        t0 = time.time()
        labels = gdb.fit(data[i], data_i[i])
        dt = time.time() - t0
        t += dt

        unique_clusters = np.unique(labels)
        print('Grid-based DBSCAN Clusters: ', unique_clusters)
        cluster_colors = list(
            plt.cm.plasma(
                np.linspace(0, 1, len(unique_clusters)+1)))  # one extra unused color at index 0 (no cluster label == 0)
        cluster_colors.append((0, 0, 0, 1))  # black for noise

        clusters = np.unique(labels)
        # Plot a fanplot
        fanplot = FanPlot(nrange=ngate, nbeam=nbeam)
        for c in clusters:
            label_mask = labels == c
            fanplot.plot(data_dict['beam'][i][label_mask], data_dict['gate'][i][label_mask], cluster_colors[c])
        plt.title('Grid-based DBSCAN fanplot\nf = %.2f    g = %d    pts_ratio = %.2f' % (f, g, pts_ratio))
        filename = '%s_f%.2f_g%d_ptRatio%.2f_scan%d_fanplot.png' % (rad_date, f, g, pts_ratio, i)
        # plt.show()
        plt.savefig(filename)
        plt.close()

        """ Velocity map """
        # Plot velocity fanplot
        fanplot = FanPlot(nrange=ngate, nbeam=nbeam)
        vel_step = 5
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

    print('Time elapsed: %.2f s' % t)

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
            labels = gdb.fit(gate, beam)

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


