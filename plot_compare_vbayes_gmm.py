import numpy as np
from cluster import *
import time
import matplotlib.pyplot as plt

def compare_gmm(data_dict, start_time, end_time, clusters=30, vbayes=False, vbayes_weight_prior=1, show=True, save=False):
    """
    Compare the fixed 30-cluster GMM algorithm with the non-fixed variational bayes GMM.
    Variational bayes GMM has a max of 10 clusters and uses the dirichlet process weight distribution.
    http://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html#sklearn.mixture.BayesianGaussianMixture

    :param data_dict:
    :param start_time:
    :param end_time:
    :return:
    """
    gate = np.hstack(data_dict['gate'])
    vel = np.hstack(data_dict['velocity'])
    wid = np.hstack(data_dict['width'])
    data_flat, data_time = flatten_data(data_dict)
    remove_close_range = gate >= 10

    # Compare accuracy vs. empirical method
    emp_gs_flg, emp_time, emp_gate = empirical(data_dict)
    num_emp = len(emp_gs_flg[remove_close_range])

    print('==================================================================')
    print(start_time)
    print('==================================================================')
    print()

    start_time_sec = time.time()
    gs_flg_gmm = gmm(data_flat, vel, wid, num_clusters=clusters)
    print('run time of GMM: {:3.2f} sec'.format(time.time() - start_time_sec))

    num_true_gmm_gs = len(np.where((gs_flg_gmm[remove_close_range] == 1) & (emp_gs_flg[remove_close_range] == 1))[0])
    num_true_gmm_is = len(np.where((gs_flg_gmm[remove_close_range] == 0) & (emp_gs_flg[remove_close_range] == 0))[0])
    accur_gmm = float(num_true_gmm_gs+num_true_gmm_is)/num_emp*100.
    print('The GS/IS identification accurary of {}-cluster GMM is {:3.2f}%'.format(clusters, accur_gmm))
    print()

    # Plot the groups
    cm = plt.cm.get_cmap('coolwarm')
    alpha = 0.2
    size = 1
    marker = 's'
    fig = plt.figure(figsize=(15,10))

    ax0 = plt.subplot(311)
    plt.scatter(emp_time[emp_gs_flg == 0], emp_gate[emp_gs_flg == 0], s=size, c='red', marker=marker, alpha=alpha,
                cmap=cm, label='GS')  # plot IS as red
    plt.scatter(emp_time[emp_gs_flg == 1], emp_gate[emp_gs_flg == 1], s=size, c='blue', marker=marker, alpha=alpha,
                cmap=cm, label='IS')  # plot GS as blue
    # plt.scatter(emp_time[emp_gs_flg == -1], emp_gate[emp_gs_flg == -1],s=size,c='blue',marker=marker, alpha=alpha)  #plot the undertermined scatter as blue
    ax0.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    # ax1.set_xlabel('Time UT')
    ax0.set_xlim([start_time, end_time])
    ax0.set_ylabel('Range gate')
    ax0.set_title('Empirical Model Results based on Burrell et al. 2015')

    import matplotlib.patches as mpatches
    blue = mpatches.Patch(color='blue', label='ground scatter')
    red = mpatches.Patch(color='red', label='ionospheric scatter')
    plt.legend(handles=[blue, red])

    ax1 = plt.subplot(312)
    plt.scatter(data_time[gs_flg_gmm == 0], gate[gs_flg_gmm == 0],s=size,c='red',marker=marker,alpha = alpha, cmap=cm)  #plot ionospheric scatter as red
    plt.scatter(data_time[gs_flg_gmm == 1], gate[gs_flg_gmm == 1],s=size,c='blue',marker=marker,alpha = alpha, cmap=cm) #plot ground scatter as blue
    ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax1.set_xlim([start_time,end_time])
    ax1.set_ylabel('Range gate')
    ax1.set_title('Standard Gaussian Mixture Model Results with an Accuracy of {:3.2f}%'.format(accur_gmm))

    if vbayes:
        start_time_sec = time.time()
        gs_flg_gmm_bayes = gmm(data_flat, vel, wid, bayes=True, weight_prior=vbayes_weight_prior)
        print('run time of 30-cluster Variational Bayes GMM: {:3.2f} sec'.format(time.time() - start_time_sec))
        num_true_gmm_bayes_gs = len(np.where((gs_flg_gmm_bayes[remove_close_range] == 1) & (emp_gs_flg[remove_close_range] == 1))[0])
        num_true_gmm_bayes_is = len(np.where((gs_flg_gmm_bayes[remove_close_range] == 0) & (emp_gs_flg[remove_close_range] == 0))[0])
        accur_gmm_bayes = float(num_true_gmm_bayes_gs+num_true_gmm_bayes_is)/num_emp*100.
        print('The GS/IS identification accurary of 30-cluster Variational Bayes GMM is {:3.2f}%'.format(accur_gmm_bayes))
        print()

        ax2 = plt.subplot(313)
        plt.scatter(data_time[gs_flg_gmm_bayes == 0], gate[gs_flg_gmm_bayes == 0],s=size,c='red',marker=marker,alpha = alpha, cmap=cm)  #plot ionospheric scatter as red
        plt.scatter(data_time[gs_flg_gmm_bayes == 1], gate[gs_flg_gmm_bayes == 1],s=size,c='blue',marker=marker,alpha = alpha, cmap=cm) #plot ground scatter as blue
        ax2.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax2.set_xlabel('Time UT')
        ax2.set_xlim([start_time,end_time])
        ax2.set_ylabel('Range gate')
        ax2.set_title('Variational Bayes Gaussian Mixture Model Results with an Accuracy of {:3.2f}% and weight prior {}'.format(accur_gmm_bayes, vbayes_weight_prior))

    if save:
        plt.savefig("bayes GMM " + start_time.__str__() + ".png")
    if show:
        plt.show()