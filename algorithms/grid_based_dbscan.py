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



if __name__ == '__main__':
    import datetime
    start_time = datetime.datetime(2018, 2, 7)
    end_time = datetime.datetime(2018, 2, 8)
    gbdb = GridBasedDBSCAN(start_time, end_time, 'cvw', scan_eps=0, load_model=False)
    print(gbdb.__dict__.keys())
    print(gbdb.runtime)
    gbdb.plot_rti(8, 'Blanchard code')
    end_time = datetime.datetime(2018, 2, 7, 23, 59)
    gbdb.plot_fanplots(start_time, end_time, show_fig=False, save_fig=True)