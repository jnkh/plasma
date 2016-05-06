from pylab import *

import numpy as np
import random
import sys
import os.path



from scipy.interpolate import interp1d,UnivariateSpline
def resample_signal(t,sig,tmin,tmax,dt):
    tt = arange(tmin,tmax,dt)
    f = UnivariateSpline(t,sig,s=0,k=1,ext=0)
    sig_interp = f(tt)
    return tt,sig_interp

def cut_signal(t,sig,tmin,tmax):
    mask = logical_and(t >= tmin,  t <= tmax)
    return t[mask],sig[mask]

def cut_and_resample_signal(t,sig,tmin,tmax,dt):
    t,sig = cut_signal(t,sig,tmin,tmax)
    return resample_signal(t,sig,tmin,tmax,dt)


def get_signals_and_ttds(signal_prepath,signals_dirs,processed_prepath,shots,
    min_times,max_times,T_max,dt,use_shots=3,recompute = False,as_array_of_shots=True):
    all_signals = []
    all_ttd = []
    use_shots = min([use_shots,len(shots)-1])
    for (j,shot_num) in enumerate(shots[:use_shots]):
        shot = shots[j]
        t_min = min_times[j]
        t_max = max_times[j]
        signals,ttd = get_signal_and_ttd(signal_prepath,signals_dirs,processed_prepath,shot,t_min,t_max,T_max,dt,recompute)
        all_signals.append(signals)
        all_ttd.append(ttd)
        print(1.0*j/use_shots)

    if as_array_of_shots:
        return array(all_signals),array(all_ttd)
    else:
        signals = vstack(all_signals)
        ttd = hstack(all_ttd)
        return signals,ttd

def get_individual_shot_file(prepath,shot_num,ext='.txt'):
    return prepath + str(shot_num) + ext 

def get_signal_and_ttd(signal_prepath,signals_dirs,processed_prepath,shot,t_min,t_max,T_max,dt,recompute = False):
    load_file_path = get_individual_shot_file(processed_prepath,shot,'.npz')
    if os.path.isfile(load_file_path) and not recompute:
        print('loading shot {}'.format(shot))
        dat = load(load_file_path)
        signals = dat['signals']
        ttd = dat ['ttd']
    else:
        print('(re)computing shot {}'.format(shot))
        signals = []
        times = []
        for (i,dirname) in enumerate(signals_dirs):
            data = loadtxt(get_individual_shot_file(signal_prepath+dirname + '/',shot))
            t = data[:,0]
            sig = data[:,1]
            tr,sigr = cut_and_resample_signal(t,sig,t_min,t_max,dt)
            signals.append(sigr)
            times.append(tr)
        signals = np.column_stack(signals)
        signals = whiten(signals)
        ttd = max(tr) - tr
        ttd = clip(ttd,0,T_max)
        ttd = log10(ttd + 1.0*dt/10)
        savez(load_file_path,signals = signals,ttd = ttd)
        print('saved shot {}'.format(shot))

    return signals,ttd

def array_to_path_and_next(arr,length,skip):
    X = []
    y = []
    i = 0
    while True:
        pred = i+length
        if pred >= len(arr):
            break
        X.append(arr[i:i+length])
        y.append(arr[i+length])
        i += skip
    X = array(X)
    X = expand_dims(X,axis=len(shape(X)))
    return X,array(y)

def array_to_path(arr,length,skip):
    X = []
    i = 0
    while True:
        pred = i+length
        if pred > len(arr):
            break
        X.append(arr[i:i+length,:])
        i += skip
    X = array(X)
    if len(shape(X)) == 1:
        X = expand_dims(X,axis=len(shape(X)))
    return X

def array_to_path_and_external_pred(arr,res,length,skip,return_sequences=False):
    assert(shape(arr)[0] == shape(res)[0])
    X = []
    y = []
    i = 0
    while True:
        pred = i+length
        if pred > len(arr):
            break
        X.append(arr[i:i+length,:])
        if return_sequences:
            y.append(res[i:i+length])
        else:
            y.append(res[i+length-1])
        i += skip
    X = array(X)
    y = array(y)
    if len(shape(X)) == 1:
        X = expand_dims(X,axis=len(shape(X)))
    if return_sequences and len(shape(y)) == 1:
        y = expand_dims(y,axis=len(shape(y)))
    return X,y

def train_test_split(x,frac):
    mask = array(range(len(x))) < frac*len(x)
    return x[mask],x[~mask]

def get_shots_and_times(shots_and_times_path):
    data = loadtxt(shots_and_times_path,npmin=1,dtype={'names':('num','timemin','timemax'),
                                                              'formats':('i4','f4','f4')})
    shots = array(zip(*data)[0])
    min_times = array(zip(*data)[1])
    max_times = array(zip(*data)[2])
    return shots,min_times,max_times

def get_shots_and_minmax_times(signal_prepath,signals_dirs,shots_and_disruption_times_path,
              current_index = 0,use_shots=-1,write_to_file=True,shots_and_minmax_times_path=None):
    shots,disruption_times = get_shots_and_disruption_times(shots_and_disruption_times_path)
    min_times = []
    max_times = []
    current_dir = signals_dirs[current_index]
    use_shots = min([use_shots,len(shots)-1])
    shots = shots[:use_shots]
    
    for (j,shot) in enumerate(shots):
        t_min,t_max = get_t_minmax(signal_prepath,signals_dirs,shot) 
        t_thresh = get_current_threshold_time(signal_prepath,current_dir,shot)
        t_disrupt = disruption_times[j]
        assert(t_thresh >= t_min)
        assert(t_disrupt <= t_max)
        assert(t_thresh < t_disrupt)
        min_times.append(t_thresh)
        max_times.append(t_disrupt)
        print(1.0*j/use_shots)
    min_times = array(min_times)
    max_times = array(max_times)
    if write_to_file:
        if shots_and_minmax_times_path == None:
            print("Not writing out file, no path given.")
        else:
            write_shots_and_minmax_times_to_file(shots,min_times,max_times,shots_and_minmax_times_path)
    return shots,array(min_times),array(max_times)

def write_shots_and_minmax_times_to_file(shots,min_times,max_times,shots_and_minmax_times_path):
    savetxt(shots_and_minmax_times_path,vstack((shots,min_times,max_times)).transpose(), fmt='%i %f %f')   
    
def read_shots_and_minmax_times_from_file(shots_and_minmax_times_path):
    data = loadtxt(shots_and_minmax_times_path,ndmin=1,dtype={'names':('num','min_times','max_times'),
                                                              'formats':('i4','f4','f4')})
    shots = array(zip(*data)[0])
    min_times = array(zip(*data)[1])
    max_times = array(zip(*data)[2])
    return shots, min_times, max_times 

def get_t_minmax(signal_prepath,signals_dirs,shot):
    t_min = -1
    t_max = Inf
    for (i,dirname) in enumerate(signals_dirs):
        data = loadtxt(signal_prepath+dirname + '/' + str(shot) + '.txt')
        t = data[:,0]
        t_min = max(t_min,t[0])
        t_max = min(t_max,t[-1])
    return t_min, t_max

def get_current_threshold_time(signal_prepath,current_dir,shot):
    current_thresh = 750000
    data = loadtxt(signal_prepath+current_dir + '/' + str(shot) + '.txt')
    t = data[:,0]
    I = data[:,1]
    assert(any(abs(I) > current_thresh))
    index_thresh = argwhere(abs(I) > current_thresh)[0][0]
    t_thresh = t[index_thresh]
    return t_thresh

def get_shots_and_disruption_times(shots_and_disruption_times_path):
    data = loadtxt(shots_and_disruption_times_path,ndmin=1,dtype={'names':('num','disrupt_times'),
                                                              'formats':('i4','f4')})
    shots = array(zip(*data)[0])
    disrupt_times = array(zip(*data)[1])
    return shots, disrupt_times
