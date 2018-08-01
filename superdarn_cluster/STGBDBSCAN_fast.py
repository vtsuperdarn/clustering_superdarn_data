"""
Grid-based DBSCAN
Author: Esther Robb

This is the fast implementation of Grid-based DBSCAN.
GBDBSCAN + Timefilter + Vel
"""

#TODO try removing the sparse matrix entirely

import numpy as np

UNCLASSIFIED = 0
NOISE = -1

class STGBDBSCAN():

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
    def _region_query(self, data, data_i, scan_i, grid_id, cluster_id, grid_labels):
        seeds = []
        hgt = self.g        #TODO should there be some rounding happening to accomidate discrete gate/wid?
        wid = self.g / (self.f * self.C[grid_id[0], grid_id[1]])        # Update f for GBDB according to current position
        ciel_hgt = int(np.ceil(hgt))
        ciel_wid = int(np.ceil(wid))

        # Check for neighbors in a box of shape ciel(2*wid), ciel(2*hgt) around the point
        g_min, g_max = max(0, grid_id[0] - ciel_hgt), min(self.ngate, grid_id[0] + ciel_hgt + 1)
        b_min, b_max = max(0, grid_id[1] - ciel_wid), min(self.nbeam, grid_id[1] + ciel_wid + 1)
        s_min, s_max = max(0, scan_i-1), min(len(data), scan_i+2)       # look at +- 1 scan
        possible_pts = 0
        for b in range(b_min, b_max):
            for g in range(g_min, g_max):
                new_id = (g, b)
                # Add the new point only if it falls within the ellipse defined by wid, hgt
                if self._in_ellipse(new_id, grid_id, hgt, wid):
                    for s in range(s_min, s_max):   # time filter
                        if self._in_ellipse(new_id, grid_id, hgt, wid):
                            possible_pts += 1
                            new_id_label = grid_labels[s][new_id]
                            # This is still less expensive than doing the ellipse calculation for every single point
                            if (new_id in data_i[s]) and (new_id_label == UNCLASSIFIED or new_id_label == NOISE or new_id_label == cluster_id):
                                rel_diff = (data[scan_i][grid_id] - data[s][new_id]) / data[scan_i][grid_id]
                                if np.abs(rel_diff) <= self.eps2:
                                #if np.abs(data[scan_i][grid_id] - data[s][new_id]) <= self.eps2:    #np.sqrt((vel1 - vel2)**2)
                                    seeds.append((s, new_id))
        return seeds, possible_pts


    def _in_ellipse(self, p, q, hgt, wid):
        return ((q[0] - p[0])**2.0 / hgt**2.0 + (q[1] - p[1])**2.0 / wid**2.0) <= 1.0        # TODO <= or <???

    # TODO bug search for this part
    def _cluster_avg(self, data, grid_labels, cluster_id):
        sum = 0.0
        size = 0.0
        for scan in range(len(grid_labels)):
            cluster_mask = grid_labels[scan] == cluster_id
            data_i = data[scan][cluster_mask]
            sum += data_i.sum()
            size += data_i.shape[1]
        return sum/size, size


    def _expand_cluster(self, data, data_i, grid_labels, scan_i, grid_id, cluster_id):
        # Find all the neighbors (including self) and the number of possible points (for grid-based DBSCAN ptRatio)
        seeds, possible_pts = self._region_query(data, data_i, scan_i, grid_id, cluster_id, grid_labels)

        k = possible_pts * self.pts_ratio
        if len(seeds) < k:
            grid_labels[scan_i][grid_id] = NOISE                    # Too small, classify as noise (may be clustered later)
            return False
        # Create a new cluster
        else:                                                       # Create a new cluster
            grid_labels[scan_i][grid_id] = cluster_id               # TODO this line is unnecessary because grid_id is in seeds
            for seed_id in seeds:                                   # Add the current point and all neighbors to this cluster
                grid_labels[seed_id[0]][seed_id[1]] = cluster_id

            # Calculate the average of the first group of points first, for later use to compare with d_eps
            cluster_avg, cluster_size = self._cluster_avg(data, grid_labels, cluster_id)

            while len(seeds) > 0:                                       # Find the neighbors of all neighbors
                current_scan, current_grid = seeds[0][0], seeds[0][1]
                results, possible_pts = self._region_query(data, data_i, current_scan, current_grid, cluster_id, grid_labels)

                k = possible_pts * self.pts_ratio
                if len(results) >= k:                                  # If this neighbor also has sufficient neighbors, add them to cluster
                    for i in range(0, len(results)):
                        result_scan, result_point = results[i][0], results[i][1]
                        # Only add a neighbor to this cluster if it hasn't already been assigned a cluster (noise is not a cluster)
                        if grid_labels[result_scan][result_point] == UNCLASSIFIED or grid_labels[result_scan][result_point] == NOISE:
                            # Compare d_eps with the existing cluster average
                            rel_diff = (cluster_avg - data[result_scan][result_point]) / cluster_avg
                            if np.abs(rel_diff) <= self.d_eps:
                            #if np.abs(cluster_avg - data[result_scan][result_point]) <= self.d_eps:
                                # Look for more neighbors only if it wasn't labelled noise previously (few neighbors)
                                if grid_labels[result_scan][result_point] != NOISE:
                                    seeds.append((result_scan, result_point))
                                grid_labels[result_scan][result_point] = cluster_id
                                # Update the cluster size and average (without looping through the whole dataset again)
                                cluster_size += 1
                                cluster_avg = (cluster_avg * (cluster_size-1) + data[result_scan][result_point]) / cluster_size
                #else:
                #    grid_labels[current_scan][current_grid] = cluster_id
                #    cluster_size += 1
                #    cluster_avg = (cluster_avg * (cluster_size-1) + data[current_scan][current_grid]) / cluster_size
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
                    if self._expand_cluster(data, data_i, grid_labels, scan_i, grid_id, cluster_id):
                        cluster_id = cluster_id + 1
                        if cluster_id == 5:
                            print('HI')

            scan_pt_labels = [grid_labels[scan_i][grid_id] for grid_id in m_i]
            point_labels.append(scan_pt_labels)
        return point_labels


def _calculate_ratio(dr, dt, i, j, r_init=0):
    r_init, dr, dt, i, j = float(r_init), float(dr), float(dt), float(i), float(j)
    cij = (r_init + dr * i) / (2.0 * dr) * (np.sin(dt * (j + 1.0) - dt * j) + np.sin(dt * j - dt * (j - 1.0)))
    return cij


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
    scans_to_use = range(2) #len(data_dict['gate']))
    """
    bx_vel = boxcox(np.abs(np.hstack(data_dict['vel'])))[0]
    values = [] #[boxcox(np.abs(data_dict['vel'][i]))[0] for i in scans_to_use] # Scan by scan boxcox.
    i = 0
    for s in np.array(data_dict['gate'])[scans_to_use]:
        values.append(bx_vel[i:i+len(s)])
        i+=len(s)
    """

    #values = [np.array(list(range(1, len(data_dict['gate'][i])+1))).astype(float) for i in scans_to_use]
    #values = [[1.0]*len(data_dict['gate'][i]) for i in scans_to_use]
    values = [np.abs(data_dict['vel'][i]) for i in scans_to_use]
    data, data_i = dict_to_csr_sparse(data_dict, ngate, nbeam, values)

    """ Grid-based DBSCAN """
    from superdarn_cluster.FanPlot import FanPlot
    import matplotlib.pyplot as plt

    # Good way to tune this one: Use only the first 200 scans of SAS 2-7-18, because they tend to get all clustered together with bad params,
    # but then there is a long period of quiet so there's time separation between that big cluster and everything else.
    dr = 45
    dtheta = 3.3
    r_init = 180
    f = 0.3
    g = 2.0
    pts_ratio = 0.5
    eps2, d_eps = 1.0, 1.0
    params = {'f' : f, 'g' : g, 'pts_ratio':pts_ratio, 'eps2':eps2, 'd_eps':d_eps}
    gdb = STGBDBSCAN(f, g, eps2, d_eps, pts_ratio, ngate, nbeam, dr, dtheta, r_init)
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
            beam_c = data_dict['beam'][i][label_mask]
            gate_c = data_dict['gate'][i][label_mask]
            fanplot.plot(beam_c, gate_c, cluster_colors[c])
            if c != -1:
                m = int(len(beam_c) / 2)  # Beam is sorted, so this is roughly the index of the median beam
                fanplot.text(str(c), beam_c[m], gate_c[m])  # Label each cluster with its cluster #

        plt.title('Grid-based DBSCAN fanplot\nparams: %s' % (params))
        filename = '%s_boxcar_f%.2f_g%d_ptRatio%.2f_2eps%.2f_deps%.2f_scan%d_fanplot.png' % (rad_date, f, g, pts_ratio, eps2, d_eps, i)
        # plt.show()
        plt.savefig(filename)
        plt.close()

        """ Velocity map """
        # Plot velocity fanplot
        fanplot = FanPlot(nrange=ngate, nbeam=nbeam)
        vel_step = 0.5
        vel_ranges = list(np.linspace(-10, 10, 41)) #list(range(-10, 10, vel_step))
        vel_ranges.insert(0, -9999)
        vel_ranges.append(9999)
        cmap = plt.cm.jet       # use 'viridis' to make this redgreen colorblind proof
        vel_colors = cmap(np.linspace(0, 1, len(vel_ranges)))
        for s in range(len(vel_ranges) - 1):
            step_mask = (values[i] >= vel_ranges[s]) & (values[i] <= (vel_ranges[s+1]))
            fanplot.plot(data_dict['beam'][i][step_mask], data_dict['gate'][i][step_mask], vel_colors[s])
        for j, beam in enumerate(data_dict['beam'][i]):
            gate = data_dict['gate'][i][j]
            fanplot.text(str(np.abs(int(values[i][j]))), beam, gate, fontsize=5)

        filename = 'bx_vel_scan%d_fanplot.png' % (i)
        fanplot.add_colorbar(vel_ranges, cmap)
        #plt.show()
        plt.savefig(filename)
        plt.close()
