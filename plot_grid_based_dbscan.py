from matplotlib.dates import date2num
import matplotlib.pyplot as plt
from superdarn_cluster.FanPlot import FanPlot
from superdarn_cluster.GridBasedDBSCAN import GridBasedDBSCAN
from superdarn_cluster.time_utils import *
import numpy as np
from superdarn_cluster.dbtools import flatten_data_11_features, read_db
import datetime as dt

start_time = dt.datetime(2018, 2, 7, 14)
end_time = dt.datetime(2018, 2, 7, 14, 15)
rad = 'sas'
db_path = "./Data/sas_GSoC_2018-02-07.db"
num_beams = 16
num_gates = 75
b = 0
data_dict = read_db(db_path, rad, start_time, end_time)
data_flat_unscaled = flatten_data_11_features(data_dict, remove_close_range=False)

gate = data_flat_unscaled[:, 1]
beam = data_flat_unscaled[:, 0]
time = data_flat_unscaled[:, 6]
time_sec = time_days_to_sec(time)
time_index = time_sec_to_index(time_sec)

data = np.column_stack((gate, beam, time_index)).T

# NOTE - these params need to change if you set remove_close_range=False. Params determined on 15min often work for longer time periods.
gdb = GridBasedDBSCAN(gate_eps=3.0, beam_eps=5.0, time_eps=30, min_pts=5, ngate=75, nbeam=16, dr=45, dtheta=3.3,
                      r_init=180)
labels = gdb.fit(data)
labels = np.array(labels)
clusters = np.unique(labels)

colors = plt.cm.plasma(np.linspace(0, 1, len(clusters)))
colors[0] = [0, 0, 0, 1]  # plot noise black
plt.figure(figsize=(16, 8))
print('clusters: ', clusters)

for b in range(num_beams):
    beam_mask = beam == b
    data_b = data[:, beam_mask]
    labels_b = labels[beam_mask]
    if not data_b.any(): continue
    for i, label in enumerate(clusters):
        plt.scatter(data_b[2, labels_b == label], data_b[0, labels_b == label], color=colors[i])
    plt.savefig("grid-based DBSCAN RTI beam " + str(b))
    plt.close()

# For each unique time unit
times_unique_dt = data_dict['datetime']
times_unique_num = [date2num(t) for t in data_dict['datetime']]
labels = np.array(labels)
print(len(times_unique_dt))

fan_colors = list(colors)
fan_colors.append((0, 0, 0, 1))
fanplot = FanPlot()
fanplot.plot_all(data_dict['datetime'], np.unique(time_index), time_index, beam, gate, labels, fan_colors,
                 base_path="grid-based dbscan ")

# TEST : epsilon division trick
from superdarn_cluster.DBSCAN import dbscan


def _calculate_ratio(dr, dt, i, j, r_init=0):
    r_init, dr, dt, i, j = float(r_init), float(dr), float(dt), float(i), float(j)
    cij = (r_init + dr * i) / (2.0 * dr) * (np.sin(dt * (j + 1.0) - dt * j) + np.sin(dt * j - dt * (j - 1.0)))
    return cij


beam_eps = 5.0 / np.array([_calculate_ratio(45, 3.3 * np.pi / 180, g, 0, r_init=180) for g in range(num_gates)])
print(np.array([_calculate_ratio(45, 3.3 * np.pi / 180, g, 0, r_init=180) for g in range(num_gates)]))
gate_eps = 3.0
time_eps = 30

beam_x = [beam[i] / beam_eps[int(gate[i])] for i in range(len(beam))]

X = np.column_stack((beam_x, gate / gate_eps, time_index / time_eps)).T

eps, min_pts = 1, 5
labels = dbscan(X, eps=eps, min_points=min_pts)
labels = np.array(labels)

fan_colors = list(colors)
fan_colors.append((0, 0, 0, 1))
fanplot = FanPlot()
fanplot.plot_all(data_dict['datetime'], np.unique(time_index), time_index, beam, gate, labels, fan_colors,
                 base_path="scaled regular DBSCAN ")

clusters = np.unique(labels)
colors = plt.cm.plasma(np.linspace(0, 1, len(clusters)))
colors[0] = [0, 0, 0, 1]  # plot noise black
plt.figure(figsize=(16, 8))
print('clusters: ', clusters)

for b in range(num_beams):
    beam_mask = beam == b
    data_b = data[:, beam_mask]
    labels_b = labels[beam_mask]
    if not data_b.any(): continue
    for i, label in enumerate(clusters):
        plt.scatter(data_b[2, labels_b == label], data_b[0, labels_b == label], color=colors[i])
    plt.savefig("scaled regular DBSCAN RTI beam " + str(b))
    plt.close()
