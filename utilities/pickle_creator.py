import pickle
from matplotlib.dates import date2num
from data_utils import read_db, get_scan_nums
import datetime as dt
from time_utils import *
import pandas as pd
import pdb

#TODO this should live in dbtools eventually? idk... i wanna be able to run this script by itself

data_dir = '../data'

# Convert a saved Algorithm model to a csv....
def model_pickle_to_csv(date, rad):
    year,month,day = date[0], date[1], date[2]
    start_time = dt.datetime(year, month, day)
    filename = "%s/%s_%s_scans.pickle" % (data_dir, rad, date_str)
    data = pickle.load(filename)
    pdb.set_trace()


def get_datestr(year, month, day):
    return '%d-%02d-%02d' % (year, month, day)

# Convert a .db file to a pickle or a csv
def convert_db(date, rad, pickle=False):
    year, month, day = date[0], date[1], date[2]
    start_time = dt.datetime(year, month, day)
    end_time = dt.datetime(year, month, day+1)
    date_str = get_datestr(year, month, day)
    db_path = "%s/%s_GSoC_%s.db" % (data_dir, rad, date_str)
    data_dict = read_db(db_path, rad, start_time, end_time)

    """ Extend features so that they are all the same length ('flatten') """
    gate = np.hstack(data_dict['gate']).astype(float)
    vel = np.hstack(data_dict['velocity'])
    wid = np.hstack(data_dict['width'])
    power = np.hstack(data_dict['power'])
    phi0 = np.hstack(data_dict['phi0'])
    elev = np.hstack(data_dict['elevation'])
    trad_gs_flg = np.hstack(data_dict['gsflg'])
    time, beam, freq, nsky, nsch = [], [], [], [], []
    num_scatter = data_dict['num_scatter']
    for i in range(len(num_scatter)):
        time.extend(date2num([data_dict['datetime'][i]] * num_scatter[i]))
        beam.extend([float(data_dict['beam'][i])] * num_scatter[i])
        freq.extend([float(data_dict['frequency'][i])] * num_scatter[i])
        nsky.extend([float(data_dict['nsky'][i])] * num_scatter[i])
        nsch.extend([float(data_dict['nsch'][i])] * num_scatter[i])
    time = np.array(time)
    beam = np.array(beam)
    freq = np.array(freq)
    nsky = np.array(nsky)
    nsch = np.array(nsch)

    """ Convert to scan-by-scan format 
    We use scan-by-scan because we want our algorithms to consider
    one scan to be one time while clustering
    It also makes fanplots easier
    """
   
    nbeam = np.max(beam) + 1
    nrang = data_dict['nrang'][0]
    scan_nums = get_scan_nums(beam)

    gate_scans = []
    beam_scans = []
    vel_scans = []
    wid_scans = []
    time_scans = []
    trad_gs_flg_scans = []
    elv_scans = []


    for s in np.unique(scan_nums):
        scan_mask = scan_nums == s
        gate_scans.append(gate[scan_mask])
        beam_scans.append(beam[scan_mask])
        vel_scans.append(vel[scan_mask])
        wid_scans.append(wid[scan_mask])
        time_scans.append(time[scan_mask])
        trad_gs_flg_scans.append(trad_gs_flg[scan_mask])
        elv_scans.append(elev[scan_mask])


    data = {'gate' : gate_scans, 'beam' : beam_scans, 'vel' : vel_scans, 'wid': wid_scans,
            'time' : time_scans, 'trad_gsflg' : trad_gs_flg_scans, 'elv': elv_scans,
            'nrang' : nrang, 'nbeam' : nbeam}
    filename = "%s/%s_%s_scans" % (data_dir, rad, date_str)

    if pickle:
        pickle.dump(data, open(filename+".pickle", 'wb'))
    else:
        df = pd.DataFrame.from_dict(data)
        df.to_csv(filename+".csv")
        



""" Get data """
rad = 'sas'

dates = [(2017, 1, 17)]#, (2017, 3, 13), (2017, 4, 4), (2017, 5, 30), (2017, 8, 20),
         #(2017, 9, 20), (2017, 10, 16), (2017, 11, 14), (2017, 12, 8), (2017, 12, 17),
         #(2017, 12, 18), (2017, 12, 19), (2018, 1, 25), (2018, 2, 7), (2018, 2, 8),
         #(2018, 3, 8), (2018, 4, 5)]


for date in dates:
    convert_db(date)
