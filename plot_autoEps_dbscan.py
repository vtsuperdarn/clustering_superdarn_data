# Based on Gaonkar & Sawant: AutoEps DBSCAN
# https://pdfs.semanticscholar.org/8cb6/fe6ad5879c8a08151481642ba92c2d603596.pdf

import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator, DataGenerator
import numpy as np
from scipy.signal import argrelextrema

def plot_k_dist(data, k):
    nbr = NearestNeighbors(n_neighbors=k+1).fit(data)
    num_pts = data.shape[0]
    distances, indices = nbr.kneighbors(data)
    distances, indices = distances[:, 1:], indices[:, 1:]   # get rid of first column, distance to self
    avg_distances = np.array([np.sum(distances[p]) / k for p in range(num_pts)])    # Avg distances for a smoother curve
    avg_distances_sorted = np.sort(avg_distances)

    plt.title('Sorted average k-th nearest neighbor distance')
    plt.ylabel('distance')
    x, y = np.array(range(num_pts)), avg_distances_sorted
    plt.scatter(x, y)
    plt.show()

    kneedle = KneeLocator(x, y)

    print(kneedle.knee)
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

data = np.column_stack((gate, beam, time_index))
min_pts_max = 10

plot_k_dist(data, k=30)
plt.ylim((0, 20))
