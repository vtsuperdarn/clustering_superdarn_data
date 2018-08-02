# Based on Gaonkar & Sawant: AutoEps DBSCAN
# https://pdfs.semanticscholar.org/8cb6/fe6ad5879c8a08151481642ba92c2d603596.pdf

import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

def find_knees(x, y):
    pass

# Simple knee alg, will only find 1 knee
# (if you want more, create an array of the dist values, and find the relative extrema)
def find_knee(x, y):
    max_dist = 0
    knee_x, knee_y = -1, -1
    y_init = y[0]
    slope = y[-1] / x[-1]

    for i in range(len(x)):
        # Data point
        px, py = x[i], y[i]
        # Point on a straight line drawn from the first data point to the last data point
        lx, ly = x[i], y_init + slope * x[i]
        dist = np.sqrt((px - lx)**2 + (py - ly)**2)
        if dist > max_dist:
            max_dist = dist
            knee_x, knee_y = px, py
    return knee_x, knee_y

def plot_k_dist(data, k, gate_eps, beam_eps, time_eps):
    nbr = NearestNeighbors(n_neighbors=k+1).fit(data)
    num_pts = data.shape[0]
    distances, indices = nbr.kneighbors(data)
    distances, indices = distances[:, 1:], indices[:, 1:]   # get rid of first column, distance to self
    avg_distances = np.array([np.sum(distances[p]) / k for p in range(num_pts)])    # Avg distances for a smoother curve
    avg_distances_sorted = np.sort(avg_distances)

    plt.figure(figsize=(16, 8))
    x, y = np.array(range(num_pts)), avg_distances_sorted
    plt.scatter(x, y)


    knee_x, knee_y = find_knee(x, y)
    print('Automatic epsilon value chosen: {}'.format(knee_y))

    plt.title('Sorted average %d-th nearest neighbor distance\ngate eps: %.1f   beam eps: %.1f   time eps: %.1f\nchosen epsilon: %.3f'
              % (k, gate_eps, beam_eps, time_eps, knee_y))
    plt.ylabel('distance')

    plt.scatter(knee_x, knee_y, s=60, color='red', label='knee')
    plt.show()

    kneedle = KneeLocator(y, x)
    kneedle.find_knee()
    kneedle.plot_knee()
    plt.show()
    kneedle.plot_knee_normalized()
    plt.show()


"""
X = np.random.normal(loc=(2, 2, 2), scale=2, size=(1000, 3))
Y = np.random.normal(loc=(-3, -3, -3), scale=10, size=(1000, 3))
data = np.vstack((X, Y))
plt.scatter(data[:, 0], data[:, 1])
plt.show()
"""
import datetime as dt
from superdarn_cluster.dbtools import read_db, flatten_data_11_features
from superdarn_cluster.time_utils import *

start_time = dt.datetime(2018, 2, 7)
end_time = dt.datetime(2018, 2, 7, 6)
rad = 'sas'
db_path = "./Data/sas_GSoC_2018-02-07.db"
data_dict = read_db(db_path, rad, start_time, end_time)
data_flat_unscaled = flatten_data_11_features(data_dict, remove_close_range=False)

gate = data_flat_unscaled[:, 1]
beam = data_flat_unscaled[:, 0]
time = data_flat_unscaled[:, 6]
time_sec = time_days_to_sec(time)
time_index = time_sec_to_index(time_sec)

k=10
gate_eps, beam_eps, time_eps = 3.0, 2.0, 40.0
data = np.column_stack((gate/gate_eps, beam/beam_eps, time_index/time_eps))
#data = np.column_stack((gate, beam, time_index))

plot_k_dist(data, k, gate_eps, beam_eps, time_eps)
plt.ylim((0, 20))
