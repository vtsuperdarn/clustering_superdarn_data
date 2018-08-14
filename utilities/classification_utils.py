import numpy as np

def blanchard_gs_flg(vel, wid, type='code'):
    med_vel = np.median(np.abs(vel))
    med_wid = np.median(np.abs(wid))
    if type == 'paper':
        return med_vel < 33.1 + 0.139 * med_wid - 0.00133 * (med_wid ** 2)  # Found in 2009 paper
    if type == 'code':
        return med_vel < 30 - med_wid * 1.0 / 3.0  # Found in RST code

def ribiero_gs_flg(vel, time):
    L = np.abs(time[-1] - time[0]) * 24
    high = np.sum(np.abs(vel) > 15.0)
    low = np.sum(np.abs(vel) <= 15.0)
    if low == 0:
        R = 1.0  # TODO hmm... this works right?
    else:
        R = high / low  # High vel / low vel ratio
    # See Figure 4 in Ribiero 2011
    if L > 14.0:
        # Addition by us
        if R > 0.15:
            return False    # IS
        else:
            return True     # GS
        # Classic Ribiero 2011
        #return True  # GS
    elif L > 3:
        if R > 0.2:
            return False
        else:
            return True
    elif L > 2:
        if R > 0.33:
            return False
        else:
            return True
    elif L > 1:
        if R > 0.475:
            return False
        else:
            return True
    # Addition by Burrell 2018 "Solar influences..."
    else:
        if R > 0.5:
            return False
        else:
            return True
    # Classic Ribiero 2011
    # else:
    #    return False
