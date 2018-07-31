"""
STDBSCAN
Author: Esther Robb

This is the fast implementation of STDBSCAN.
DBSCAN + Timefilter + Vel
"""

#TODO this has a lot of remnants of GBDBSCAN that are not needed

import numpy as np

UNCLASSIFIED = 0
NOISE = -1

class STDBSCAN():

    def __init__(self, eps1, eps2, d_eps, pts_ratio, ngate, nbeam):
        self.eps1 = eps1
        self.eps2 = eps2
        self.d_eps = d_eps
        self.pts_ratio = pts_ratio
        self.ngate = ngate
        self.nbeam = nbeam


    def _region_query(self, data, data_i, scan_i, grid_id, cluster_id, grid_labels):
        seeds = []
        box_size = int(np.ceil(self.eps1))
        # Check for neighbors in a box of shape ciel(2*wid), ciel(2*hgt) around the point
        g_min, g_max = max(0, grid_id[0] - box_size), min(self.ngate, grid_id[0] + box_size + 1)
        b_min, b_max = max(0, grid_id[1] - box_size), min(self.nbeam, grid_id[1] + box_size + 1)
        s_min, s_max = max(0, scan_i-1), min(len(data), scan_i+2)       # look at +- 1 scan
        possible_pts = 0
        for b in range(b_min, b_max):
            for g in range(g_min, g_max):
                new_id = (g, b)
                # Add the new point only if it falls within the ellipse defined by wid, hgt
                if self._in_range(new_id, grid_id):
                    for s in range(s_min, s_max):   # time filter
                        if self._in_range(new_id, grid_id):
                            possible_pts += 1
                            # It could use data_i to test if it's a valid point! More robust.
                            # In fact... do we even need the sparse matrix in that case? LOL
                            new_id_label = grid_labels[s][new_id]
                            # This might still be less expensive than doing the ellipse calculation for every single point
                            if (new_id in data_i[s]) and (new_id_label == UNCLASSIFIED or new_id_label == NOISE or new_id_label == cluster_id):
                                rel_diff = (data[scan_i][grid_id] - data[s][new_id]) / data[scan_i][grid_id]
                                if np.abs(rel_diff) <= self.eps2:    #np.sqrt((vel1 - vel2)**2)
                                #if np.abs(data[scan_i][grid_id] - data[s][new_id]) <= self.eps2:    #np.sqrt((vel1 - vel2)**2)
                                    seeds.append((s, new_id))
        return seeds, possible_pts


    def _in_range(self, p, q):
        return np.sqrt((q[0] - p[0])**2 + (q[1]-p[1])**2) <= self.eps1

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

        k = self.pts_ratio # possible_pts * self.pts_ratio
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

                k = self.pts_ratio # possible_pts * self.pts_ratio
                if len(results) >= k:                                  # If this neighbor also has sufficient neighbors, add them to cluster
                    for i in range(0, len(results)):
                        result_scan, result_point = results[i][0], results[i][1]
                        # Only add a neighbor to this cluster if it hasn't already been assigned a cluster (noise is not a cluster)
                        if grid_labels[result_scan][result_point] == UNCLASSIFIED or grid_labels[result_scan][result_point] == NOISE:
                            # Compare d_eps with the existing cluster average
                            rel_diff = (cluster_avg - data[result_scan][result_point]) / cluster_avg
                            if np.abs(rel_diff) <= self.d_eps:
                            #if np.abs(cluster_avg - data[result_scan][result_point]) <= self.d_eps:
                                #print(cluster_avg)
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

            scan_pt_labels = [grid_labels[scan_i][grid_id] for grid_id in m_i]
            point_labels.append(scan_pt_labels)
        return point_labels


def dict_to_csr_sparse(data_dict, ngate, nbeam, values, gate_scale = 1.0, beam_scale = 1.0):
    from scipy import sparse
    gate = data_dict['gate']
    beam = data_dict['beam']
    nscan = len(values)
    data = []
    data_i = []
    for i in range(nscan):
        m = sparse.csr_matrix((values[i], (gate[i] / gate_scale, beam[i] / beam_scale)), shape=(ngate, nbeam))
        m_i = list(zip(np.array(gate[i]).astype(int), np.array(beam[i]).astype(int)))
        data.append(m)
        data_i.append(m_i)
    return data, data_i

# TODO what about when there's no data... this assumes there is always data
# TODO can these scripts be converted to *not* use the CSR matrix, but just do a smarter search, and still be fast?

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
    scans_to_use = range(3) #len(data_dict['gate']))
    values = [np.abs(data_dict['vel'][i]).astype(int) for i in scans_to_use]
    #values = [np.array(list(range(1, len(data_dict['gate'][i])+1))).astype(float) for i in scans_to_use]
    #values = [[1.0]*len(data_dict['gate'][i]) for i in scans_to_use]
    data, data_i = dict_to_csr_sparse(data_dict, ngate, nbeam, values)

    for value in values[0].astype(int):
        print('%s;' % value, end='')

    """ Grid-based DBSCAN """
    from superdarn_cluster.FanPlot import FanPlot
    import matplotlib.pyplot as plt

    # Solid params board: f=0.2 0.3 0.4, g=1 2 3
    # Good way to tune this one: Use only the first 200 scans of SAS 2-7-18, because they tend to get all clustered together with bad params,
    # but then there is a long period of quiet so there's time separation between that big cluster and everything else.
    dr = 45
    dtheta = 3.3
    r_init = 180
    eps1 = 3.0
    pts_ratio = 6
    eps2, d_eps = 20.0, 20.0
    gdb = STDBSCAN(eps1, eps2, d_eps, pts_ratio, ngate, nbeam)
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


