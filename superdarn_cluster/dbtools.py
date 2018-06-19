import json
import sqlite3
import numpy as np
from matplotlib.dates import date2num
from sklearn import preprocessing

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

#TODO rewrite this function it is getting ugly with all the scaling and filtering
def flatten_data(data_dict, extras=False, scale=True, center=False, transform=False, remove_close_range=False):
    """
    Note: This will not work for scale = True and transform = True.

    Helper function to :
    > convert a dictionary from the database to a NumPy array
    > normalize values
    """

    gate = np.hstack(data_dict['gate'])
    vel = np.hstack(data_dict['velocity'])
    wid = np.hstack(data_dict['width'])
    power = np.hstack(data_dict['power'])
    phi0 = np.hstack(data_dict['phi0'])
    time, beam = [], []

    num_scatter = data_dict['num_scatter']
    for i in range(len(num_scatter)):
        time.extend(date2num([data_dict['datetime'][i]] * num_scatter[i]))
        beam.extend([float(data_dict['beam'][i])] * num_scatter[i])

    time = np.array(time)
    beam = np.array(beam)

    if remove_close_range:
        filter = gate > 10
        beam = beam[filter]
        gate = gate[filter]
        vel = vel[filter]
        wid = wid[filter]
        power = power[filter]
        phi0 = phi0[filter]
        time = time[filter]

    if transform:
        """ Gaussify some non-Gaussian features """
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

    if not scale:
        data = np.column_stack((beam, gate, vel, wid, power, phi0, time))

    else:
        #data = np.column_stack((beam_scaled, gate_scaled, vel_scaled, wid_scaled,
        #                        power_scaled, phi0_scaled, time_scaled))
        data = np.column_stack((vel_scaled, wid_scaled, phi0_scaled))

    # feature names= ['beam', 'gate', 'vel', 'wid', 'power', 'phi0', 'time']

    if extras:
        if remove_close_range:
            return data, beam, gate, vel, wid, power, phi0, time, filter
        return data, beam, gate, vel, wid, power, phi0, time

    return data, time

def flatten_data_11_features(data_dict, scaled=False, remove_close_range=False):
    gate = np.hstack(data_dict['gate']).astype(float)
    vel = np.hstack(data_dict['velocity'])
    wid = np.hstack(data_dict['width'])
    power = np.hstack(data_dict['power'])
    phi0 = np.hstack(data_dict['phi0'])
    elev = np.hstack(data_dict['elevation'])
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

    if remove_close_range:
        filter = gate > 10
        gate = gate[filter]
        vel = vel[filter]
        wid = wid[filter]
        power = power[filter]
        beam = beam[filter]
        time = time[filter]
        phi0 = phi0[filter]
        freq = freq[filter]
        nsky = nsky[filter]
        nsch = nsch[filter]
        elev = elev[filter]

    # Scale s.t. variance is 1 and mean is 0
    gate_scaled = preprocessing.scale(gate)
    vel_scaled = preprocessing.scale(vel)
    wid_scaled = preprocessing.scale(wid)
    power_scaled = preprocessing.scale(power)
    beam_scaled = preprocessing.scale(beam)
    time_scaled = preprocessing.scale(time)
    phi0_scaled = preprocessing.scale(phi0)
    freq_scaled = preprocessing.scale(freq)
    nsky_scaled = preprocessing.scale(nsky)
    nsch_scaled = preprocessing.scale(nsch)
    elev_scaled = preprocessing.scale(elev)

    if not scaled:
        return np.column_stack((beam, gate, vel, wid,
                                power, freq, time,
                                phi0, elev, nsky, nsch))


    else:
        return np.column_stack((beam_scaled, gate_scaled,vel_scaled,wid_scaled,
                                power_scaled,freq_scaled, time_scaled,
                                phi0_scaled, elev_scaled, nsky_scaled, nsch_scaled))