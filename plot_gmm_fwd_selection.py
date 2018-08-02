from superdarn_cluster.dbtools import *
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import pandas as pd
import os

#feature_names = np.array(['gate', 'vel', 'wid', 'power', 'freq', 'time', 'phi0', 'elev', 'nsky', 'nsch'])
#feature_names = np.array(['vel', 'wid', 'power', 'phi0', 'elev', 'nsky', 'nsch'])
#feature_names = np.array(['gate', 'time', 'elev'])
feature_names = np.array(['vel', 'wid', 'phi0'])
#feature_names = np.array(['PC'+str(i) for i in range(7)])


def forward_selection(data_flat, num_clusters=range(2,6), num_features=7, old_features=[], models=[]):
    # Features to select from:
    features = [f for f in range(num_features) if f not in old_features]
    # Recursive end condition
    if not features:
        return models

    aics, bics = [], []
    convergence_flag = []
    estimators = []

    for fi, feature in enumerate(features):
        test_features = list(old_features)
        test_features.append(feature)
        test_data = data_flat[:, test_features]
        aics.append([])
        bics.append([])
        convergence_flag.append([])
        estimators.append([])

        for ci, nc in enumerate(num_clusters):
            estimator = GaussianMixture(n_components=nc, covariance_type='full', max_iter=500,
                                        random_state=0, n_init=5, init_params='kmeans')
            estimator.fit(test_data)
            estimators[fi].append(estimator)
            aic = estimator.aic(test_data)
            bic = estimator.bic(test_data)

            aics[fi].append(aic)
            bics[fi].append(bic)

        plt.plot(num_clusters, bics[fi])
        plt.title('num clusters vs. bic, ' + str(feature_names[test_features]))
        plt.savefig('num clusters vs. bic, ' + str(feature_names[test_features]) + '.png')
        plt.close()

        #plt.plot(test_data)
        #plt.show()

    aics = np.array(aics)
    bics = np.array(bics)

    # Don't let it choose a discrete feature for the first recursion
    #if old_features == []:
    #    features = features[1:]
    #    aics = aics[1:, :]
    #    bics = bics[1:, :]


    best_bic = np.inf
    best_bic_i = (-1, -1)
    for fi in range(len(features)):
        if best_bic > bics[fi, 0]:
            best_bic = bics[fi, 0]
            best_bic_i = (fi, 0)

        for ci in range(1, len(num_clusters)):
            ratio = bics[fi, ci] / bics[fi, ci-1]
            if ci <= 15:
                tolerance = 0.025
            else:
                tolerance = 0.05
            converged = ratio > (1 - tolerance) and ratio < (1 + tolerance) and ci >= 1
            if best_bic > bics[fi, ci] and not converged:
                best_bic = bics[fi, ci]
                best_bic_i = (fi, ci)


    #best_aic = np.unravel_index(aics.argmin(), aics.shape)
    #best_bic_i = np.unravel_index(bics.argmin(), bics.shape)

    best_features = list(old_features)      # Make a copy of old_features
    best_features.append(features[best_bic_i[0]])
    best_num_clusters = num_clusters[best_bic_i[1]]
    best_estimator = estimators[best_bic_i[0]][best_bic_i[1]]

    model = {"features": best_features, "num_clusters": best_num_clusters,
             "bic": bics[best_bic_i], 'aic': aics[best_bic_i], 'estimator': best_estimator}

    aics_df = pd.DataFrame(data=aics, index=feature_names[features], columns=num_clusters)
    bics_df = pd.DataFrame(data=bics, index=feature_names[features], columns=num_clusters)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print('AIC for model', model)
        print(aics_df)
        print()
        print('BIC for model', model)
        print(bics_df)
        print()

    models.append(model)
    return forward_selection(data_flat, num_clusters=num_clusters, num_features=num_features, old_features=best_features, models=models)


def plot_model_selection(data_dict):
    # Iterate thru each feature to find which one has the best BIC score
    # Pick that one, then iterate thru all combos with that feature
    # Etc
    #data_flat, time = flatten_data(data_dict, remove_close_range=True)
    data_flat = flatten_data_11_features(data_dict)
    time = data_flat[:, 6]
    vel = data_flat[:, 2]
    gate = data_flat[:, 1]
    range_max = data_dict['nrang'][0]
    start_time = dt.datetime(2018, 2, 7, 12)
    end_time = dt.datetime(2018, 2, 7, 14)
    # beam, gate, vel, wid, power, freq, time, phi0, elev, nsky, nsch
    features_i = [2,3,7]
    data_flat_features = data_flat[:, features_i]    # remove 'beam' feature
    #scaled_data = data_flat - np.mean(data_flat, axis=0)
    #pca = PCA(n_components=7)
    #pca.fit(scaled_data)
    #pca_data = pca.transform(scaled_data)
    #models = forward_selection(data_flat, num_clusters=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30,35,40], num_features=10)
    models = forward_selection(data_flat_features, num_features=len(features_i), num_clusters=[2,3,4,5,6,7])#,8,9,10,11,12,13,14,15])#20,25,30,35,40,45,50], num_features=10)

    from superdarn_cluster.utilities import plot_gmm_clusters, plot_feature_pairs_by_cluster
    print("############ MODELS ###############")
    for i, model in enumerate(models):

        estimator = model['estimator']

        model_data = data_flat_features[:, model['features']]
        estimator.fit(model_data)
        cluster_membership = estimator.predict(model_data)
        dir = 'model ' + str(i+1) + " clusters"
        os.makedirs(dir)
        dir = dir + '/'
        plot_gmm_clusters(cluster_membership, data_flat, time, gate, vel, ['beam', 'gate', 'vel', 'wid', 'power', 'freq', 'time', 'phi0', 'elev', 'nsky', 'nsch'],
                          range_max, start_time, end_time, num_clusters=model['num_clusters'], base_path=dir)
        plot_feature_pairs_by_cluster(model_data, estimator, feature_names[model['features']], base_path=dir)

        model['features'] = list(feature_names[model['features']])
        print(i+1, model)

    bics = [model['bic'] for model in models]
    model_num = range(1, len(models)+1)

    plt.plot(model_num, bics)
    plt.ylabel('BIC')
    plt.xlabel('Model #')
    plt.show()
    return


if __name__ == '__main__':
    import datetime as dt

    skip = []
    start_time = dt.datetime(2018, 2, 7, 12)
    end_time = dt.datetime(2018, 2, 7, 14)

    rad = 'sas'
    db_path = "./Data/sas_GSoC_2018-02-07.db"

    for i in range(1):
        if i in skip:
            continue

        s = start_time + dt.timedelta(i)
        e = start_time + dt.timedelta(i + 1)
        data = read_db(db_path, rad, start_time, end_time, beam=0)
        plot_model_selection(data)
