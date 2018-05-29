from cluster import *
import datetime as dt

def compare_accuracy_over_time(rad, start_time, num_days, skip=[]):
    """
    Generate a graph to compare the accuracy of traditional method, kmeans, and GMM over time.
    :param rad: name of database, e.g. "SAS"
    :param start_time: first day
    :param num_days: number of days
    :param skip: list of days to skip (integers, start_time = day 0)
    """
    accur_tras = []
    accur_kmeanss = []
    accur_gmms = []
    times = []

    # No-good data in sas.db:  09/17-09/19
    for i in range(num_days):
        # Skip over bad-data days
        if i in skip:
            continue

        data_dict = read_db(rad, start_time+dt.timedelta(i), start_time+dt.timedelta(i+1))
        emp_gs_flg, emp_time, emp_gate = empirical(data_dict)
        trad_gs_flg = traditional(data_dict)
        data_flat, _ = flatten_data(data_dict)
        vel = np.hstack(data_dict['velocity'])
        wid = np.hstack(data_dict['width'])
        km_gs_flag = kmeans(data_flat, vel, wid)
        gmm_gs_flg = gmm(data_flat, vel, wid)

        num_true_trad_gs = len(np.where(((trad_gs_flg == 1) | (trad_gs_flg == -1)) & (emp_gs_flg == 1))[0])
        num_true_trad_is = len(np.where(((trad_gs_flg == 0)) & (emp_gs_flg == 0))[0])

        num_emp = len(emp_gs_flg)
        accur_tra = float(num_true_trad_gs + num_true_trad_is) / num_emp * 100.

        num_true_kmeans_gs = len(np.where((km_gs_flag == 1) & (emp_gs_flg == 1))[
                                     0])  # Assuming the GS is the cluster with minimum median velocity
        num_true_kmeans_is = len(np.where((km_gs_flag == 0) & (emp_gs_flg == 0))[0])
        accur_kmeans = float(num_true_kmeans_gs + num_true_kmeans_is) / num_emp * 100.

        num_true_gmm_gs = len(np.where((gmm_gs_flg == 1) & (emp_gs_flg == 1))[
                                  0])  # Assuming the GS is the cluster with minimum median velocity
        num_true_gmm_is = len(np.where((gmm_gs_flg == 0) & (emp_gs_flg == 0))[0])
        accur_gmm = float(num_true_gmm_gs + num_true_gmm_is) / num_emp * 100.

        accur_tras.append(accur_tra)
        accur_kmeanss.append(accur_kmeans)
        accur_gmms.append(accur_gmm)
        times.append(start_time+dt.timedelta(i))

    times = date2num(times)
    print(accur_tras)
    print(accur_kmeans)
    print(accur_gmms)

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    plt.plot(times, accur_tras, c='black', label='Traditional Model')
    plt.plot(times, accur_kmeanss, c='blue', label='Kmeans')
    plt.plot(times, accur_gmms, c='red', label='GMM')
    ax.set_ylim(0, 100)
    ax.legend(loc='best')
    ax.set_ylabel('Accuracy %')
    ax.set_xlabel('Date')
    ax.set_xlim([start_time, (start_time + dt.timedelta(num_days))])
    ax.xaxis.set_major_formatter(DateFormatter('%m-%d'))

    plt.show()