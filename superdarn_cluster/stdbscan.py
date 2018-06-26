"""
Author: Lucas Miguel Simões Ponce
Source: https://github.com/eubr-bigsea/py-st-dbscan
"""

from datetime import timedelta
from geopy.distance import great_circle

def st_dbscan(df, spatial_threshold, temporal_threshold, min_neighbors):
    """
    Python st-dbscan implementation.
    INPUTS:
        df={o1,o2,...,on} Set of objects
        spatial_threshold = Maximum geographical coordinate (spatial) distance
        value
        temporal_threshold = Maximum non-spatial distance value
        min_neighbors = Minimun number of points within Eps1 and Eps2 distance
    OUTPUT:
        C = {c1,c2,...,ck} Set of clusters
    """
    cluster_label = 0
    noise = -1
    unmarked = 777777
    stack = []

    # initialize each point with unmarked
    df['cluster'] = unmarked

    # for each point in database
    for index, point in df.iterrows():
        if df.loc[index]['cluster'] == unmarked:
            # Iterates thru all data entries again
            neighborhood = retrieve_neighbors(index, df, spatial_threshold,
                                              temporal_threshold)

            if len(neighborhood) < min_neighbors:
                df.set_value(index, 'cluster', noise)
            else:  # found a core point
                cluster_label += 1
                # assign a label to core point
                df.set_value(index, 'cluster', cluster_label)

                # assign core's label to its neighborhood
                for neig_index in neighborhood:
                    df.set_value(neig_index, 'cluster', cluster_label)
                    stack.append(neig_index)  # append neighborhood to stack

                # find new neighbors from core point neighborhood
                while len(stack) > 0:
                    current_point_index = stack.pop()
                    new_neighborhood = retrieve_neighbors(
                        current_point_index, df, spatial_threshold,
                        temporal_threshold)

                    # current_point is a new core
                    if len(new_neighborhood) >= min_neighbors:
                        for neig_index in new_neighborhood:
                            neig_cluster = df.loc[neig_index]['cluster']
                            if all([neig_cluster != noise,
                                    neig_cluster == unmarked]):
                                # TODO: verify cluster average
                                # before add new point
                                df.set_value(neig_index, 'cluster',
                                             cluster_label)
                                stack.append(neig_index)
    return df


def retrieve_neighbors(index_center, df, spatial_threshold, temporal_threshold):
    neigborhood = []

    center_point = df.loc[index_center]

    # TODO this
    # filter by time 
    min_time = center_point['time'] - temporal_threshold
    max_time = center_point['time'] + temporal_threshold
    df = df[(df['time'] >= min_time) & (df['time'] <= max_time)]

    # filter by distance
    for index, point in df.iterrows():
        if index != index_center:
            distance = great_circle(
                (center_point['x'], center_point['y']),
                (point['x'], point['y'])).meters
            if distance <= spatial_threshold:
                neigborhood.append(index)

    return neigborhood


if __name__ == '__main__':
    import numpy as np
    from sklearn.cluster import DBSCAN
    from sklearn import metrics

    # #############################################################################
    # Get data
    from superdarn_cluster.dbtools import flatten_data_11_features, read_db
    import datetime as dt

    start_time = dt.datetime(2018, 2, 7, 12)
    end_time = dt.datetime(2018, 2, 7, 14)
    rad = 'sas'
    db_path = "../Data/sas_GSoC_2018-02-07.db"
    b = 0
    data_dict = read_db(db_path, rad, start_time, end_time)
    data_flat_unscaled = flatten_data_11_features(data_dict, remove_close_range=True)


    import matplotlib.pyplot as plt
    feature_names = ['beam', 'gate', 'vel', 'wid', 'power', 'freq', 'time', 'phi0', 'elev', 'nsky', 'nsch']

    gate = data_flat_unscaled[:, 1]
    power = data_flat_unscaled[:, 4]
    beam = data_flat_unscaled[:, 0]
    vel = data_flat_unscaled[:, 2]

    time = data_flat_unscaled[:, 6]
    #time = (time - np.floor(time)) * 24 * 60 * 60 / secs_per_measurement

    # What matters for scaling this is the size of each step between these (discrete) measurements.
    # If you want to connect things within 1 range gate and 1 beam, do no scaling and set eps ~= 1.1
    # If you want to connect things within 6 time measurements, scale it so that 6 * dt = 1 and eps ~= 1.1
    # Time has some gaps in between each scan of 16 beams, so epsilon should be large enough
    scaled_time = (time - time[0]) #(time - np.floor(time)) * 24 * 60 * 60
    uniq_time = np.sort(np.unique(scaled_time))
    shifted_time = np.roll(uniq_time, -1)
    dt = np.min((shifted_time - uniq_time)[:-1])
    integer_time = scaled_time / dt
    scaled_time = scaled_time / dt
    # Divide by variance and center mean at 0
    scaled_gate = gate
    scaled_beam = beam

    X = np.column_stack((beam, gate, integer_time))
    print(X.shape)
    print(integer_time[:20])

    # ~~ ST-DBSCAN ~~
    from superdarn_cluster.stdbscan import st_dbscan
    import pandas as pd

    space_eps = 15
    time_eps = 25
    min_pts = 2
    df = pd.DataFrame(X, columns=['x', 'y', 'time'])
    # slooooow
    df = st_dbscan(df, spatial_threshold=space_eps, temporal_threshold=time_eps, min_neighbors=min_pts)

    labels = np.array(df['cluster'])
    clusters = np.unique(labels)
    print(clusters)

    # #############################################################################
    # Plot result

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    range_max = data_dict['nrang'][0]

    fig = plt.figure(figsize=(16, 8))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)
        xy = X[class_member_mask, :]
        plt.plot(df['x'], df['y'], '.', color=tuple(col), markersize=3)

    plt.title(
        'Beam %d \n Clusters: %d   Eps1: %.2f   Eps2: %.2f   MinPts: %d ' % (b, len(clusters), space_eps, time_eps, min_pts))
    plt.show()
