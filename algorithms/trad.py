import sys
sys.path.insert(0,'..')
from algorithms.algorithm import Traditional


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
        vel_max=200
        vel_step=25
    elif rad == 'cvw':
        vel_max=100
        vel_step=10
    else:
        print('Cant use that radar')
        exit()

    print(rad)
    print(dates)

    for date in dates:
        start_time = date
        end_time = date + datetime.timedelta(days=1)
        gmm = Traditional(start_time, end_time, rad)
        gmm.plot_rti('*', 'Blanchard code', vel_max=vel_max, vel_step=vel_step, show_fig=False, save_fig=True)
        #dbgmm.plot_fanplots(start_time, end_time, vel_max=100, vel_step=10, show=False, save=True)
