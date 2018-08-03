import numpy as np

def time_days_to_index(time_days):
    return time_sec_to_index(time_days_to_sec(time_days))

def time_sec_to_index(time_sec):
    uniq_time = np.sort(np.unique(time_sec))
    shifted_time = np.roll(uniq_time, -1)       # circular left shift to compute all the time deltas
    dt = np.min((shifted_time - uniq_time)[:-1])
    # dt = np.median((shifted_time - uniq_time)[:-1])
    index_time = time_sec / dt
    return index_time

def time_days_to_sec(time):
    time_sec = np.round((time - time[0]) * 24 * 60 * 60)
    #np.round(np.unique(time_sec).reshape((-1, 1)))
    return time_sec