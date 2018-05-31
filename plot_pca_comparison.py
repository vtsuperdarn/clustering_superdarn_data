import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from superdarn_cluster.dbtools import *
from pandas import DataFrame

def plot_pca_density(data_dict, save=True, num_clusters=6):
    gate = np.hstack(data_dict['gate'])
    vel = np.hstack(data_dict['velocity'])
    wid = np.hstack(data_dict['width'])
    data_flat, time = flatten_data(data_dict)
    feature_names = ['beam', 'gate', 'vel', 'wid', 'power', 'phi0', 'time']
    num_bins = [16, len(np.unique(gate)), 1000, 1000, 1000, 1000, 1000]


    # Great StackOverflow thread on recovering features from PCA:
    # https://stackoverflow.com/questions/22984335/recovering-features-names-of-explained-variance-ratio-in-pca-with-sklearn
    pca = PCA(n_components=num_clusters)
    pca.fit(data_flat)
    components = DataFrame(pca.components_, columns=feature_names)
    print("PCA components 0-" + str(num_clusters-1) + " with feature correlations")
    print("(Each PCA component is a linear combination of all features,")
    print("with component 0 having the highest importance)")
    print(components.to_string())
    data_flat_pca = pca.transform(data_flat)

    for i in range(data_flat.shape[1]):
        feat = data_flat[:, i]
        feat_name = feature_names[i]

        ax0 = plt.subplot(111)
        ax0.set_xlabel(feat_name)
        ax0.set_ylabel('density')
        ax0.hist(feat, bins=num_bins[i], color='orange')

        plt.savefig("no pca " + feat_name + ".png")
        plt.close()

    for i in range(data_flat_pca.shape[1]):
        feat_pca = data_flat_pca[:, i]
        feat_name = feature_names[i]

        ax0 = plt.subplot(111)
        ax0.set_title("PCA")
        ax0.set_xlabel(feat_name)
        ax0.set_ylabel('density')
        ax0.hist(feat_pca, bins=50, color='blue')

        plt.savefig("pca" + str(i) + ".png")
        plt.close()


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
        data = read_db(db_path, rad, s, e)
        #compare_pca(data, num_clusters=2, save=False)
        plot_pca_density(data, save=False)