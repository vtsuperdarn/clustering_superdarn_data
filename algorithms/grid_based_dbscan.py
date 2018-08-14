import sys
sys.path.insert(0,'..')
from algorithms.algorithm import GridBasedDBAlgorithm

class GridBasedDBSCAN(GridBasedDBAlgorithm):

    def __init__(self, start_time, end_time, rad,
                 f=0.2, g=1, pts_ratio=0.3,
                 dr=45, dtheta=3.24, r_init=180,
                 scan_eps=1,
                 load_model=False,
                 save_model=False):
        super().__init__(start_time, end_time, rad,
                         {'f': f,
                          'g': g,
                          'pts_ratio': pts_ratio,
                          'scan_eps': scan_eps,     # for scan by scan set = 0, for timefilter set >= 1
                          'dr': dr,
                          'dtheta': dtheta,
                          'r_init': r_init},
                         load_model=load_model)
        if not load_model:
            data, data_i = self._get_gbdb_data_matrix(self.data_dict)
            clust_flg, self.runtime = self._gbdb(data, data_i)
            self.clust_flg = self._1D_to_scanxscan(clust_flg)
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
        gbdb = GridBasedDBSCAN(start_time, end_time, rad, scan_eps=1, load_model=False, save_model=True)
        gbdb.plot_rti('*', threshold, vel_max=vel_max, vel_step=vel_step, show_fig=False, save_fig=True)
        #dbgmm.plot_fanplots(start_time, end_time, vel_max=100, vel_step=10, show=False, save=True)
