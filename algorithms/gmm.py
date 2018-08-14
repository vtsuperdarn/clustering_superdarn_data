import sys
sys.path.insert(0,'..')
from algorithms.algorithm import GMMAlgorithm
import numpy as np

class GMM(GMMAlgorithm):
    """
    GMM for SuperDARN data.
    """
    def __init__(self, start_time, end_time, rad,
                 n_clusters=10, cov='full',
                 features=['beam', 'gate', 'time', 'vel', 'wid'],
                 BoxCox=False,
                 load_model=False,
                 save_model=False):
        super().__init__(start_time, end_time, rad,
                         {'n_clusters' : n_clusters,
                          'cov': cov,
                          'features': features,
                          'BoxCox': BoxCox},
                         load_model=load_model)

        if not load_model:
            clust_flg, self.runtime = self._gmm(self._get_gmm_data_array())
            self.clust_flg = self._1D_to_scanxscan(clust_flg)
            print(np.unique(np.hstack(self.clust_flg)))
        if save_model:
            self._save_model()

import sys

if __name__ == '__main__':

    import datetime
    from datetime import datetime as dt

    dates = [dt(2017, 1, 17), dt(2017, 3, 13), dt(2017, 4, 4), dt(2017, 5, 30), dt(2017, 8, 20),
             dt(2017, 9, 20), dt(2017, 10, 16), dt(2017, 11, 14), dt(2017, 12, 8), dt(2017, 12, 17),
             dt(2017, 12, 18), dt(2017, 12, 19), dt(2018, 1, 25), dt(2018, 2, 7), dt(2018, 2, 8),
             dt(2018, 3, 8), dt(2018, 4, 5)]
    rad = sys.argv[1]

    if rad == 'sas':
        threshold = 'Blanchard code'
        vel_max=200
        vel_step=25
    elif rad == 'cvw':
        threshold = 'Ribiero'
        vel_max=100
        vel_step=10
    else:
        print('Cant use that radar')
        exit()

    print(rad)
    print(dates)
    print(threshold)

    for date in dates:
        start_time = date
        end_time = date + datetime.timedelta(days=1)
        gmm = GMM(start_time, end_time, rad,
                           load_model=False, save_model=True, BoxCox=True)
        gmm.plot_rti('*', threshold, vel_max=vel_max, vel_step=vel_step, show_fig=False, save_fig=True)
        #dbgmm.plot_fanplots(start_time, end_time, vel_max=100, vel_step=10, show=False, save=True)
