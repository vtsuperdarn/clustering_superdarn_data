import json
import sqlite3
import numpy as np
import os

def read_db(db_path, rad, start_time, end_time, beam='*'):
    """
    Read from a SQL database
    db_path: path to .db file
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


def _monotonic(vec):
    if len(vec) < 2:
        return True
    return all(x <= y for x, y in zip(vec[:-1], vec[1:])) \
           or all(x >= y for x, y in zip(vec[:-1], vec[1:]))


# TODO make this work for other satellite modes (what are the other satellite modes?)
# see radDataRead -> readScan
def get_scan_nums(beams_flat):
    """
    Figure out scan number label for each data point in beams_flat.

    NOTE: Only works on normal mode - beam # always increasing/decreasing in one scan!

    :param beams_flat: list/array of beams from flatten_data array
    :return: Integer scan number for each data point in beams_flat
    """
    scan = 0
    i = 0
    scan_nums = np.zeros(len(beams_flat)).astype(int)
    while i < len(beams_flat):
        scan_i = []
        j = 0
        while i + j + 1 <= len(beams_flat):
            new_scan = list(range(i, i+j+1))
            if _monotonic(beams_flat[new_scan]):
                scan_i = new_scan
                j += 1
            else:
                break
        scan_nums[scan_i] = scan
        scan += 1
        i += j
    return scan_nums


def get_data_dict_path(day, rad, data_dir='../data'):
    """
    Get the path to a pickled data dictionary for a certain day
    :param day: datetime object
    :param rad: radar code, such as 'sas'
    :param data_dir: the directory where data is stored
    :return:
    """
    # Get data_dict from pickle file
    this_dir = os.path.abspath(os.path.dirname(__file__))
    date_str = day.strftime('%Y-%m-%d')
    return "%s/%s/%s_%s_scans.pickle" % (this_dir, data_dir, rad, date_str)

