from superdarn_cluster.dbtools import *
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import pandas as pd

feature_names = np.array(['gate', 'vel', 'wid', 'power', 'freq', 'time', 'phi0', 'elev', 'nsky', 'nsch'])
#feature_names = np.array(['PC'+str(i) for i in range(7)])

def forward_selection(data_flat, num_clusters=range(2,6), num_features=7, old_features=[], models=[]):
    # Features to select from:
    features = [f for f in range(num_features) if f not in old_features]
    # Recursive end condition
    if not features:
        return models

    aics, bics = [], []

    for fi, feature in enumerate(features):
        test_features = list(old_features)
        test_features.append(feature)
        test_data = data_flat[:, test_features]
        aics.append([])
        bics.append([])

        for ci, nc in enumerate(num_clusters):
            estimator = GaussianMixture(n_components=nc, covariance_type='full', max_iter=500,
                                        random_state=0, n_init=5, init_params='kmeans')
            estimator.fit(test_data)
            aic = estimator.aic(test_data)
            bic = estimator.bic(test_data)

            aics[fi].append(aic)
            bics[fi].append(bic)

        plt.plot(test_data)
        plt.show()

    aics = np.array(aics)
    bics = np.array(bics)
    # Don't let it choose a discrete feature for the first recursion
    if old_features == []:
        features = features[1:]
        aics = aics[1:, :]
        bics = bics[1:, :]

    best_aic = np.unravel_index(aics.argmin(), aics.shape)
    best_bic = np.unravel_index(bics.argmin(), bics.shape)

    best_features = list(old_features)      # Make a copy of old_features
    best_features.append(features[best_bic[0]])
    best_num_clusters = num_clusters[best_bic[1]]

    model = {"features": best_features, "num_clusters": best_num_clusters, "bic": bics[best_bic], 'aic': aics[best_bic]}

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
    data_flat = data_flat[:, 1:]    # remove 'beam' feature
    #scaled_data = data_flat - np.mean(data_flat, axis=0)
    #pca = PCA(n_components=7)
    #pca.fit(scaled_data)
    #pca_data = pca.transform(scaled_data)
    #models = forward_selection(data_flat, num_clusters=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30,35,40], num_features=10)
    models = forward_selection(data_flat, num_clusters=[2, 5, 10], num_features=10)

    print("############ MODELS ###############")
    for i, model in enumerate(models):
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
    start_time = dt.datetime(2018, 2, 7)
    rad = 'cvw'
    db_path = "./Data/cvw_GSoC_2018-02-07.db"

    for i in range(1):
        if i in skip:
            continue

        s = start_time + dt.timedelta(i)
        e = start_time + dt.timedelta(i + 1)
        data = read_db(db_path, rad, s, e, beam=12)
        plot_model_selection(data)
