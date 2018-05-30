from cluster import empirical, read_db, gmm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier


def plot_forest_importances(data_dict, labels='empirical'):
    """
    Based on this method:
    http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#sphx-glr-auto-examples-ensemble-plot-forest-importances-py

    Note: The results of this vary A LOT based on whether you use GMM, or empirical, or pass the forest the unscaled data, or the scaled.
          The one point of agreement is that range gate and time are always important.
          However, we need to use a more robust method of model selection, since this is so variable.

    Trains a forest to rank feature importance
    :param data_dict: dictionary from read_db
    :param labels: 'gmm' or 'empirical', which classification method to use as "ground truth"
    """

    gate = np.hstack(data_dict['gate'])
    vel = np.hstack(data_dict['velocity'])
    wid = np.hstack(data_dict['width'])
    power = np.hstack(data_dict['power'])
    phi0 = np.hstack(data_dict['phi0'])
    time, beam, freq, nsky, nsch = [], [], [], [], []

    num_scatter = data_dict['num_scatter']
    for i in range(len(num_scatter)):
        time.extend(date2num([data_dict['datetime'][i]] * num_scatter[i]))
        beam.extend([float(data_dict['beam'][i])] * num_scatter[i])
        freq.extend([float(data_dict['frequency'][i])] * num_scatter[i])
        nsky.extend([float(data_dict['nsky'][i])] * num_scatter[i])
        nsch.extend([float(data_dict['nsch'][i])] * num_scatter[i])

    time = np.array(time)
    beam = np.array(beam)
    freq = np.array(freq)
    nsky = np.array(nsky)
    nsch = np.array(nsch)

    # Scale s.t. variance is 1 and mean is 0
    gate_scaled = preprocessing.scale(gate)
    vel_scaled = preprocessing.scale(vel)
    wid_scaled = preprocessing.scale(wid)
    power_scaled = preprocessing.scale(power)
    beam_scaled = preprocessing.scale(beam)
    time_scaled = preprocessing.scale(time)
    phi0_scaled = preprocessing.scale(phi0)
    freq_scaled = preprocessing.scale(freq)
    nsky_scaled = preprocessing.scale(nsky)
    nsch_scaled = preprocessing.scale(nsch)

    data_flat = np.column_stack((beam_scaled, gate_scaled,vel_scaled,wid_scaled,
                            power_scaled,freq_scaled, time_scaled,
                            phi0_scaled, nsky_scaled, nsch_scaled))

    data_flat_unscaled = np.column_stack((beam, gate, vel, wid,
                            power, freq, time,
                            phi0, nsky, nsch))

    feat_names = ["beam", "gate", "vel", "wid", "power", "freq", "time", "phi0", "nsky", "nsch"]


    if labels == 'empirical':
        gs_flg = empirical(data_dict)
    elif labels == 'gmm':
        gs_flg = gmm(data_flat, vel, wid)
    else:
        print('Bad label: ', labels)
        return

    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=500, random_state=0)

    forest.fit(data_flat_unscaled, gs_flg)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    sorted_feat_names = []
    for f in indices:
        print("feature %s (%f)" % (feat_names[f], importances[f]))
        sorted_feat_names.append(feat_names[f])

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(data_flat.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(data_flat.shape[1]), sorted_feat_names)
    plt.xlim([-1, data_flat.shape[1]])
    plt.show()

if __name__ == '__main__':
    import datetime as dt

    skip = []
    start_time = dt.datetime(2018, 2, 7)
    days = 1
    rad = 'sas'
    db_path = "./Data/sas_GSoC_2018-02-07.db"

    for i in range(days):
        if i in skip:
            continue

        s = start_time + dt.timedelta(i)
        e = start_time + dt.timedelta(i + 1)
        data = read_db(db_path, rad, s, e)
        plot_forest_importances(data)