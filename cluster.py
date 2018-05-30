import json
import sqlite3
import numpy as np
from matplotlib.dates import date2num
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.decomposition import PCA
import pickle


def read_db(db_path, rad, start_time, end_time, beam='*'):
    """
    Read from a SQL database
    rad: 3-letter radar appreviation, like 'sas' or 'cvw' (case insensitive
    start_time: datetime object, start time
    end_time: datetime object, end time
    """
    rad = rad.lower()
    # make a db connection
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = conn.cursor()

    if beam != '*':
        command = "SELECT * FROM {tb}\
                    WHERE time BETWEEN '{stm}' AND '{etm}'\
                    AND beam = {beam}\
                    ORDER BY time". \
            format(tb="sd_table_"+rad, beam=beam, stm=start_time, etm=end_time)
    else:
        command = "SELECT * FROM {tb}\
                    WHERE time BETWEEN '{stm}' AND '{etm}'\
                    ORDER BY time". \
            format(tb="sd_table_"+rad, stm=start_time, etm=end_time)

    cur.execute(command)
    rws = cur.fetchall()
    if not rws:
        return False

    data_dict = dict()

    #We'll use the following parameters (or features) to do the clustering or predictions
    data_dict['datetime'] = [x[18] for x in rws]                #datetime
    data_dict['beam'] = [x[0] for x in rws]                     #beam number  (dimentionless)
    data_dict['nrang'] = [x[10] for x in rws]				    #number of range gates
    data_dict['num_scatter'] = [x[13] for x in rws]             #number of scatter return in one beam at one scan (dimentionless)
    data_dict['frequency'] = [x[5] for x in rws]                #radar transmited frequency [MHz]
    data_dict['nsky'] = [x[12] for x in rws]                    #sky noise level
    data_dict['nsch'] = [x[11] for x in rws]                    #freq search noise level
    data_dict['power'] = [json.loads(x[15]) for x in rws]       #return signal power [dB]
    data_dict['velocity'] = [json.loads(x[19]) for x in rws]    #Doppler velocity [m/s]
    data_dict['width'] = [json.loads(x[22]) for x in rws]       #spectral width   [m/s]
    data_dict['gate'] = [json.loads(x[6]) for x in rws]         #range gate (dimentionless)
    data_dict['gsflg'] = [json.loads(x[7]) for x in rws]
    data_dict['hop'] = [json.loads(x[8]) for x in rws]
    data_dict['elevation'] = [json.loads(x[2]) for x in rws]    #elevation angle [degree]
    data_dict['phi0'] = [json.loads(x[14]) for x in rws]        #phi0 for calculation of elevation angle
    return data_dict


def flatten_data(data_dict, extras=False, transform=False):
    """
    Helper function to :
    > convert a dictionary from the database to a NumPy array
    > normalize values
    """

    gate = np.hstack(data_dict['gate'])
    vel = np.hstack(data_dict['velocity'])
    wid = np.hstack(data_dict['width'])
    power = np.hstack(data_dict['power'])
    elev = np.hstack(data_dict['elevation'])
    phi0 = np.hstack(data_dict['phi0'])
    time, beam = [], []

    num_scatter = data_dict['num_scatter']
    for i in range(len(num_scatter)):
        time.extend(date2num([data_dict['datetime'][i]] * num_scatter[i]))
        beam.extend([float(data_dict['beam'][i])] * num_scatter[i])

    time = np.array(time)
    beam = np.array(beam)

    if transform:
        """ Gaussian preprocessing """
        # Assuming feature order: ['beam', 'gate', 'vel', 'wid', 'power', 'phi0', 'time']
        g_gate = gate ** 2.0      # RG = RG^2
        g_wid = np.sign(wid) * np.log(np.abs(wid))
        g_vel = np.sign(vel) * np.log(np.abs(vel))
        g_power = power ** 1.5

        gate_scaled = preprocessing.scale(g_gate)
        vel_scaled = preprocessing.scale(g_vel)
        wid_scaled = preprocessing.scale(g_wid)
        power_scaled = preprocessing.scale(g_power)

    else:
        gate_scaled = preprocessing.scale(gate)
        vel_scaled = preprocessing.scale(vel)
        wid_scaled = preprocessing.scale(wid)
        power_scaled = preprocessing.scale(power)

    # Scale s.t. variance is 1 and mean is 0
    beam_scaled = preprocessing.scale(beam)
    time_scaled = preprocessing.scale(time)
    phi0_scaled = preprocessing.scale(phi0)
    #freq_scaled = preprocessing.scale(freq)
    #nsky_scaled = preprocessing.scale(nsky)
    #nsch_scaled = preprocessing.scale(nsch)


    """
    data = np.column_stack((beam_scaled, gate_scaled,vel_scaled,wid_scaled,
                            power_scaled,freq_scaled, time_scaled,
                            phi0_scaled, nsky_scaled, nsch_scaled))
    """
    data = np.column_stack((beam_scaled, gate_scaled, vel_scaled, wid_scaled,
                            power_scaled, phi0_scaled, time_scaled))

    if extras:
        return data, beam, gate, vel, wid, power, phi0, time

    #data = np.column_stack((beam, gate, vel, wid, power, elev, freq, time, phi0, nsky, nsch))
    return data, time


def empirical(data_dict):
    """
    gs_hops = [1.0, 2.0, 3.0]
    is_hops = [0.5, 1.5, 2.5]

    emp_gate = np.hstack(data_dict['gate'])
    emp_time, emp_gsflg = [], []
    emp_num_scatter = data_dict['num_scatter']

    for i in range(len(emp_num_scatter)):
        emp_time.extend(date2num([data_dict['datetime'][i]]*emp_num_scatter[i]))
        for j in range(len(data_dict['hop'][i])):
            if data_dict['hop'][i][j] in is_hops:
                emp_gsflg.append(0)
            elif data_dict['hop'][i][j] in gs_hops:
                emp_gsflg.append(1)

    return np.array(emp_gsflg)
    """
    return np.hstack(data_dict['gsflg'])

def traditional(data_dict):
    #vel = np.hstack(data_dict['velocity'])
    #wid = np.hstack(data_dict['width'])
    #gs_trad = vel < (33.1 + 0.139 * wid - 0.00133 * (wid ** 2))
    #return gs_trad
    return np.abs(np.hstack(data_dict['gsflg']))


# TODO add a bool param that plots the results by cluster. Could be useful, but which dimensions to use? Time vs. gate makes sense, or time vs. gate vs. vel
def gmm(data_flat, vel, wid,
        num_clusters=30,  vel_threshold=15,
        bayes=False, weight_prior=1, pca=False,
        cluster_identities=False,
        make_pickle="", use_pickle=""):

    pickle_dir = "./GMMPickles/"
    if use_pickle != "":
        use_picklefile = pickle_dir + use_pickle
        cluster_labels = pickle.load(use_picklefile)

    else:
        # source
        # http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html#sphx-glr-auto-examples-mixture-plot-gmm-covariances-py
        if pca:
            pca = PCA(n_components=num_clusters) #TODO what should this be? see what works best
            pca.fit(data_flat)
            data_flat = pca.transform(data_flat)

        if not bayes:
            estimator = GaussianMixture(n_components=num_clusters,
                                        covariance_type='full', max_iter=500,
                                        random_state=0, n_init=5, init_params = 'kmeans')
        elif bayes:
            # Params to fine tune:
            # Weight concentration prior - determines number of clusters chosen, 0.1 1 100 1000 no big difference.
            # Concentration prior type - determines the distribution type
            # Max iter - maybe this will need increasing
            # Mean precision prior
            # Mean prior - should this be 0 since we centered the data there? seems to work well!!!
            #
            # n-init - does okay even on just 1 initialization, will increasing it be better?
            # num_clusters - 5 seems like way too few, when you look at clusters of data in just space and time.
            #                See what looks reasonable. I think it'd be good to get a real view of the clustering in space and time,
            #                individual clsuters and not just GS / IS.
            estimator = BayesianGaussianMixture(n_components=num_clusters,
                                                covariance_type='full', max_iter=500,
                                                random_state=0, n_init=5, init_params='kmeans',
                                                weight_concentration_prior=weight_prior,
                                                weight_concentration_prior_type='dirichlet_process')


        estimator.fit(data_flat)
        if bayes:
            print('weights for variational bayes')
            print(estimator.weights_)
            print(np.sum(estimator.weights_ > 0.01), 'clusters used (weight > 1%)')
            print('converged?')
            print(estimator.converged_)
            print('number of iterations to converge')
            print(estimator.n_iter_)
            print('lower bound on likelihood')
            print(estimator.lower_bound_)
            print('weight prior')
            print(weight_prior)


        cluster_labels = estimator.predict(data_flat)

        if make_pickle != "":
            make_picklefile = pickle_dir + use_pickle
            pickle.dump(cluster_labels, make_picklefile)

    gs_class_gmm = []
    is_class_gmm = []
    median_vels_gmm = np.zeros(num_clusters)
    median_wids_gmm = np.zeros(num_clusters)
    for i in range(num_clusters):
        median_vels_gmm[i] = np.median(np.abs(vel[cluster_labels == i]))
        median_wids_gmm[i] = np.median(wid[cluster_labels == i])
        #print median_vels_gmm[i]
        if median_vels_gmm[i] > vel_threshold:
            is_class_gmm.append(i)
        else:
            gs_class_gmm.append(i)

    gs_flg_gmm = []
    for i in cluster_labels:
        if i in gs_class_gmm:
            gs_flg_gmm.append(1)
        elif i in is_class_gmm:
            gs_flg_gmm.append(0)

    if cluster_identities:
        clusters = [np.where(cluster_labels == i)[0] for i in range(num_clusters)]
        return np.array(gs_flg_gmm), clusters, median_vels_gmm
    else:
        return np.array(gs_flg_gmm)


def kmeans(data_flat, vel, wid, num_clusters=5,  vel_threshold = 10., num_init = 50, pca=False):
    # Do Principal Component Analysis to reduce dimensionality
    if pca:
        pca = PCA()
        pca.fit(data_flat)
        data_flat = pca.transform(data_flat)

    # Fit kmeans++
    Z_kmeans = KMeans(init = 'k-means++', n_clusters = num_clusters, n_init = num_init).fit_predict(data_flat)
    median_vels_kmeans = np.zeros(num_clusters)
    median_wids_kmeans = np.zeros(num_clusters)
    gs_class_kmeans = []
    is_class_kmeans = []

    for i in range(num_clusters):
        # Find median velocity and spectral width
        median_vels_kmeans[i] = np.median(np.abs(vel[Z_kmeans == i]))
        median_wids_kmeans[i] = np.median(wid[Z_kmeans == i])

        #print median_vels_kmeans[i]
        # Split as GS or IS by median velocity
        if median_vels_kmeans[i] > vel_threshold:
            is_class_kmeans.append(i)
        else:
            gs_class_kmeans.append(i)
    # Create a set of data labels to compare to ground truth, where 1 is GS and 0 is IS
    gs_flg_kmeans = []
    for i in Z_kmeans:
        if i in gs_class_kmeans:
            gs_flg_kmeans.append(1)
        elif i in is_class_kmeans:
            gs_flg_kmeans.append(0)

    gs_flg_kmeans = np.array(gs_flg_kmeans)
    return gs_flg_kmeans


