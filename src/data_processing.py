from __future__ import print_function

from pylab import *

import numpy as np
import random
import sys
import os.path
from scipy.cluster.vq import whiten
from scipy.interpolate import interp1d,UnivariateSpline

from os import listdir,remove
from os.path import isfile, join


def clean_shots_lists(shots_lists_dir):
    paths = [join(shots_lists_dir, f) for f in listdir(shots_lists_dir) if isfile(join(shots_lists_dir, f))]
    for path in paths:
        clean_shots_list(path)


def append_to_filename(path,to_append):
    ending_idx = path.rfind('.')
    new_path = path[:ending_idx] + to_append + path[ending_idx:]
    return new_path


def clean_shots_list(path):
    data = loadtxt(path)
    ending_idx = path.rfind('.')
    new_path = append_to_filename(path,'_clear')
    if len(shape(data)) < 2:
        #nondisruptive
        nd_times = -1.0*ones_like(data)
        data_two_column = vstack((data,nd_times)).transpose()
        savetxt(new_path,data_two_column,fmt = '%d %f')
        print('created new file: {}'.format(new_path))
        print('deleting old file: {}'.format(path))
        os.remove(path)


def resample_signal(t,sig,tmin,tmax,dt):
    order = argsort(t)
    t = t[order]
    sig = sig[order]
    tt = arange(tmin,tmax,dt)
    f = UnivariateSpline(t,sig,s=0,k=1,ext=0)
    sig_interp = f(tt)
    if(any(isnan(sig_interp))): print("signals contains nan")
    if(any(t[1:] - t[:-1] <= 0)):
	print("non increasing")
	idx = where(t[1:] - t[:-1] <= 0)[0][0]
	print(t[idx-10:idx+10])

    return tt,sig_interp

def cut_signal(t,sig,tmin,tmax):
    mask = logical_and(t >= tmin,  t <= tmax)
    return t[mask],sig[mask]

def cut_and_resample_signal(t,sig,tmin,tmax,dt):
    t,sig = cut_signal(t,sig,tmin,tmax)
    return resample_signal(t,sig,tmin,tmax,dt)


def get_signals_and_ttds(signal_prepath,signals_dirs,processed_prepath,shots,
    min_times,max_times,disruptive,T_max,dt,use_shots=3,recompute = False,as_array_of_shots=True):
    all_signals = []
    all_ttd = []
    use_shots = min([use_shots,len(shots)])
    for (j,shot_num) in enumerate(shots[:use_shots]):
        shot = shots[j]
        t_min = min_times[j]
        t_max = max_times[j]
        disruptive_shot = disruptive[j]
        signals,ttd = get_signal_and_ttd(signal_prepath,signals_dirs,processed_prepath,shot,t_min,t_max,disruptive_shot,T_max,dt,recompute)
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

def get_signal_and_ttd(signal_prepath,signals_dirs,processed_prepath,shot,t_min,t_max,disruptive,T_max,dt,recompute = False):
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
        if disruptive:
            ttd = max(tr) - tr
            ttd = clip(ttd,0,T_max)
        else:
            ttd = T_max*np.ones_like(tr)
        ttd = log10(ttd + 1.0*dt/10)
        savez(load_file_path,signals = signals,ttd = ttd)
        print('saved shot {}'.format(shot))

    return signals,ttd




def get_signals_and_times_from_file(shot,t_disrupt,conf):
    valid = True
    t_min = -1
    t_max = Inf
    t_thresh = -1
    signals = []
    times = []
    signal_prepath = conf['paths']['signal_prepath']
    signals_dirs = conf['paths']['signals_dirs']
    current_index = conf['data']['current_index']
    current_thresh = conf['data']['current_thresh']
    for (i,dirname) in enumerate(signals_dirs):
        data = loadtxt(get_individual_shot_file(signal_prepath+dirname + '/',shot))
        t = data[:,0]
        sig = data[:,1]
        t_min = max(t_min,t[0])
        t_max = min(t_max,t[-1])
        if i == current_index:
            if not (any(abs(sig) > current_thresh)):
                valid = False
                print('Shot {} does not exceed current threshold... invalid.'.format(shot))
            else:
                index_thresh = argwhere(abs(sig) > current_thresh)[0][0]
                t_thresh = t[index_thresh]
        signals.append(sig)
        times.append(t)
    if not valid:
        t_thresh = t_min
    assert(t_thresh >= t_min)
    assert(t_disrupt <= t_max)
    if t_disrupt >= 0:
        assert(t_thresh < t_disrupt)
        t_max = t_disrupt
    t_min = t_thresh

    return signals,times,t_min,t_max,t_thresh,valid



def cut_and_resample_signals(times,signals,t_min,t_max,is_disruptive,conf,standard_deviations = None):
    dt = conf['data']['dt']
    T_max = conf['data']['T_max']

    #resample signals
    signals_processed = []
    assert(len(signals) == len(times) and len(signals) > 0)
    tr = 0
    for i in range(len(signals)):
        tr,sigr = cut_and_resample_signal(times[i],signals[i],t_min,t_max,dt)
        signals_processed.append(sigr)

    signals = signals_processed
    signals = np.column_stack(signals)
    if standard_deviations is None:
        signals = whiten(signals)
        print('warning: whitening each signal individually')
    else:
        print('STD GLOBAL')
        print(standard_deviations)
        print('STD')
        print(std(signals,0))
        print('MEAN')
        print(mean(signals,0))
        print('MEAN ABS')
        print(mean(np.abs(signals),0))
        print('MIN')
        print(np.min(signals,0))
        print('MAX')
        print(np.max(signals,0))
        signals /= standard_deviations
    if is_disruptive:
        ttd = max(tr) - tr
        ttd = np.clip(ttd,0,T_max)
    else:
        ttd = T_max*np.ones_like(tr)
    ttd = log10(ttd + 1.0*dt/10)
    return signals,ttd

def get_normalizations_for_signals(times,signals,t_min,t_max,is_disruptive,conf):
    dt = conf['data']['dt']
    T_max = conf['data']['T_max']

    #resample signals
    signals_processed = []
    assert(len(signals) == len(times) and len(signals) > 0)
    tr = 0
    for i in range(len(signals)):
        tr,sigr = cut_and_resample_signal(times[i],signals[i],t_min,t_max,dt)
        signals_processed.append(sigr)

    signals = signals_processed
    signals = np.column_stack(signals)
    standard_deviations = std(signals,0)
    return standard_deviations



def preprocess_all_shots(conf):
    shot_files = conf['paths']['shot_files'] + conf['paths']['shot_files_test']
    shot_list_dir = conf['paths']['shot_list_dir']
    use_shots = conf['data']['use_shots']
    return preprocess_all_shots_from_files(conf,shot_list_dir,shot_files,use_shots)


def preprocess_all_shots_from_files(conf,shot_list_dir,shot_files,use_shots):
    shots,disruption_times = get_multiple_shots_and_disruption_times(shot_list_dir,shot_files)

    standard_deviations = preprocess_data_whitener(conf)

    dt = conf['data']['dt']
    processed_prepath = conf['paths']['processed_prepath']
    recompute = conf['data']['recompute']
    use_shots = min(use_shots,len(shots))
    used_shots = []
    disruptive = []
    indices = np.random.choice(arange(len(shots)),size=use_shots,replace=False)
    num_processed = 0
    for j in indices:
        num_processed += 1
        print('({}/{}): '.format(num_processed,use_shots))
        shot = shots[j]
        load_file_path = get_individual_shot_file(processed_prepath,shot,'.npz')
        if recompute or not os.path.isfile(load_file_path):
            print('(re)computing shot {}'.format(shot))
            t_disrupt = disruption_times[j]
            is_disruptive =  t_disrupt >= 0
          #get minmax times
            signals,times,t_min,t_max,t_thresh,valid = get_signals_and_times_from_file(shot,t_disrupt,conf) 
            #cut and resample
            signals,ttd = cut_and_resample_signals(times,signals,t_min,t_max,is_disruptive,conf,standard_deviations)

            savez(load_file_path,signals = signals,ttd = ttd,is_disruptive=is_disruptive,valid=valid)
            print('saved shot {}'.format(shot))
        else:
            dat = load(load_file_path)
            valid = dat['valid']
            is_disruptive = dat['is_disruptive']
        if valid:
            print('valid')
            used_shots.append(shot)
            disruptive.append(bool_to_int(is_disruptive))
        else:
            print('Warning: shot {} not valid, omitting'.format(shot))
    print('Omitted {} shots of {} total.'.format(use_shots - len(used_shots),use_shots))
    print('{}/{} disruptive shots'.format(sum(disruptive),len(disruptive)))
    return array(used_shots), array(disruptive)


def preprocess_data_whitener(conf):
    #only use training shots here!! "Don't touch testing shots"
    shot_files = conf['paths']['shot_files']# + conf['paths']['shot_files_test']
    shot_list_dir = conf['paths']['shot_list_dir']
    use_shots = max(20,int(round(0.1*conf['data']['use_shots'])))
    return preprocess_data_whitener_from_files(conf,shot_list_dir,shot_files,use_shots)


def preprocess_data_whitener_from_files(conf,shot_list_dir,shot_files,use_shots):
    shots,disruption_times = get_multiple_shots_and_disruption_times(shot_list_dir,shot_files)

    dt = conf['data']['dt']
    normalizer_path = conf['paths']['normalizer_path']
    recompute = conf['data']['recompute_normalization']
    use_shots = min(use_shots,len(shots))
    used_shots = []
    disruptive = []
    indices = np.random.choice(arange(len(shots)),size=use_shots,replace=False)
    num_processed = 0
    standard_deviations = []#zeros(conf['data']['num_signals'])
    mins = []
    maxs = []
    num_disruptive = 0
    if recompute or not os.path.isfile(normalizer_path):
        for j in indices:
            print('({}/{}): '.format(num_processed,use_shots))
            shot = shots[j]
            print('(re)computing shot {} for normalization'.format(shot))
            t_disrupt = disruption_times[j]
            is_disruptive =  t_disrupt >= 0
          #get minmax times
            signals,times,t_min,t_max,t_thresh,valid = get_signals_and_times_from_file(shot,t_disrupt,conf) 
            #cut and resample
            standard_deviations_curr = get_normalizations_for_signals(times,signals,t_min,t_max,is_disruptive,conf)
            if valid:
                standard_deviations.append(standard_deviations_curr)
                num_processed += 1
                num_disruptive += (1 if is_disruptive else 0)

        standard_deviations = np.row_stack(standard_deviations)
        standard_deviations = np.median(standard_deviations,0)
        np.savez(normalizer_path,standard_deviations = standard_deviations,num_processed = num_processed,num_disruptive = num_disruptive)
        print('saving normalization data from {} shots, {} disruptive'.format(num_processed,num_disruptive))
    else:
        dat = load(normalizer_path)
        standard_deviations = dat['standard_deviations']
        num_processed = dat['num_processed']
        num_disruptive = dat['num_disruptive']
        print('loading normalization data from {} shots, {} disruptive'.format(num_processed,num_disruptive))
    return standard_deviations


def bool_to_int(predicate):
    return 1 if predicate else 0

def time_is_disruptive(t):
    return 1 if t >= 0 else 0 

def times_are_disruptive(ts):
    return array([time_is_disruptive(t) for t in ts])

def load_shot_as_X_y(conf,shot,verbose=False):
    dt = conf['data']['dt']
    length = conf['model']['length']
    skip = conf['model']['skip']
    processed_prepath = conf['paths']['processed_prepath']
    remapper = conf['data']['ttd_remapper']

    all_signals = []
    all_ttd = []
    disruptive = []
    used_shots = []
    
    load_file_path = get_individual_shot_file(processed_prepath,shot,'.npz')
    assert(os.path.isfile(load_file_path))
    if verbose:
        print('loading shot {}'.format(shot))
    dat = load(load_file_path)
    signals = dat['signals']
    ttd = dat ['ttd']
    is_disruptive = dat ['is_disruptive']
    valid = dat['valid']
    assert(valid)

    ttd = remapper(ttd,conf['data']['T_warning'])
    X,y = array_to_path_and_external_pred(signals,ttd,length,skip)
    return  X,y


def load_shots_as_X_y(conf,shots):
    X,y = zip(*[load_shot_as_X_y(conf,shot) for shot in shots])
    return vstack(X),hstack(y)


def load_or_preprocess_all_shots_from_files(conf,shot_list_dir,shot_files):
   
    shots,disruption_times = get_multiple_shots_and_disruption_times(shot_list_dir,shot_files)

    dt = conf['data']['dt']
    processed_prepath = conf['paths']['processed_prepath']
    recompute = conf['data']['recompute']
    use_shots = min([conf['data']['use_shots'],len(shots)])

    all_signals = []
    all_ttd = []
    disruptive = []
    used_shots = []
    
    for (j,shot) in enumerate(shots[:use_shots]):
        shot = shots[j]
        load_file_path = get_individual_shot_file(processed_prepath,shot,'.npz')
        if os.path.isfile(load_file_path) and not recompute:
            print('loading shot {}'.format(shot))
            dat = load(load_file_path)
            signals = dat['signals']
            ttd = dat ['ttd']
            is_disruptive = dat ['is_disruptive']
            valid = dat['valid']
        else:
            print('(re)computing shot {}'.format(shot))
            t_disrupt = disruption_times[j]
            is_disruptive =  t_disrupt >= 0
          #get minmax times
            signals,times,t_min,t_max,t_thresh,valid = get_signals_and_times_from_file(shot,t_disrupt,conf) 
            #cut and resample
            signals,ttd = cut_and_resample_signals(times,signals,t_min,t_max,is_disruptive,conf)

            savez(load_file_path,signals = signals,ttd = ttd,is_disruptive=is_disruptive,valid = valid)
            print('saved shot {}'.format(shot))

        if valid:
            disruptive.append(1 if is_disruptive else 0)
            all_signals.append(signals)
            all_ttd.append(ttd)
            used_shots.append(shot)
            print(1.0*j/use_shots)
        else:
            print('Shot {} not valid, omitting.'.format(shot))
    return array(all_signals),array(all_ttd),array(disruptive),array(used_shots)


def load_or_preprocess_all_shots(conf):
    shot_files = conf['paths']['shot_files']
    shot_list_dir = conf['paths']['shot_list_dir']
    return load_all_shots_from_files(conf,shot_list_dir,shot_files)
    

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

def train_test_split(x,frac,shuffle_data=False):
    mask = array(range(len(x))) < frac*len(x)
    if shuffle_data:
        shuffle(mask)
    return x[mask],x[~mask]

def train_test_split_all(x,frac,shuffle_data=True):
    groups = []
    length = len(x[0])
    mask = array(range(length)) < frac*length
    if shuffle_data:
        shuffle(mask)
    for item in x:
        groups.append((item[mask],item[~mask]))
    return groups


def load_all_shots_and_minmax_times(shot_list_dir,shot_files,signal_prepath,signals_dirs,current_index,use_shots,recompute_minmax):
    all_shots = []
    all_min_times = []
    all_max_times = []
    all_disruptive = []
    for shot_filename in shot_files:
        shot_path = join(shot_list_dir,shot_filename)
        shot_and_minmax_times_path = append_to_filename(shot_path,'_minmax_times')
        if os.path.isfile(shot_and_minmax_times_path) and not recompute_minmax:
            print('minmax previously generated for {}, reading file'.format(shot_path))
            shots,min_times,max_times,disruptive = read_shots_and_minmax_times_from_file(shot_and_minmax_times_path)
        else:
            print('generating minmax for {}'.format(shot_path))
            shots,min_times,max_times,disruptive = get_shots_and_minmax_times(signal_prepath,signals_dirs,shot_path,
                   current_index,use_shots,True,shot_and_minmax_times_path)
        all_shots.append(shots)
        all_min_times.append(min_times)
        all_max_times.append(max_times)
        all_disruptive.append(disruptive)
    return hstack(all_shots), hstack(all_min_times), hstack(all_max_times), hstack(all_disruptive)



# def get_shots_and_times(shots_and_times_path):
#     data = loadtxt(shots_and_times_path,npmin=1,dtype={'names':('num','timemin','timemax'),
#                                                               'formats':('i4','f4','f4')})
#     shots = array(zip(*data)[0])
#     min_times = array(zip(*data)[1])
#     max_times = array(zip(*data)[2])
#     return shots,min_times,max_times

def get_shots_and_minmax_times(signal_prepath,signals_dirs,shots_and_disruption_times_path, 
              current_index = 0,use_shots=-1,write_to_file=True,shots_and_minmax_times_path=None):
    shots,disruption_times = get_shots_and_disruption_times(shots_and_disruption_times_path)
    min_times = []
    max_times = []
    disruptive = []
    current_dir = signals_dirs[current_index]
    use_shots = min([use_shots,len(shots)])
    shots = shots[:use_shots]
    
    for (j,shot) in enumerate(shots):
        t_min,t_max = get_t_minmax(signal_prepath,signals_dirs,shot) 
        t_thresh = get_current_threshold_time(signal_prepath,current_dir,shot)
        t_disrupt = disruption_times[j]
        assert(t_thresh >= t_min)
        assert(t_disrupt <= t_max)
        if t_disrupt > 0:
            assert(t_thresh < t_disrupt)
        min_times.append(t_thresh)
        if t_disrupt < 0:
            disruptive.append(0)
            max_times.append(t_max)
        else:
            disruptive.append(1)
            max_times.append(t_disrupt)
        print(1.0*j/use_shots)
    min_times = array(min_times)
    max_times = array(max_times)
    disruptive = array(disruptive)
    if write_to_file:
        if shots_and_minmax_times_path == None:
            print("Not writing out file, no path given.")
        else:
            write_shots_and_minmax_times_to_file(shots,min_times,max_times,disruptive,shots_and_minmax_times_path)
    return shots,min_times,max_times,disruptive

def write_shots_and_minmax_times_to_file(shots,min_times,max_times,disruptive,shots_and_minmax_times_path):
    savetxt(shots_and_minmax_times_path,vstack((shots,min_times,max_times,disruptive)).transpose(), fmt='%i %f %f %i')   
    
def read_shots_and_minmax_times_from_file(shots_and_minmax_times_path):
    data = loadtxt(shots_and_minmax_times_path,ndmin=1,dtype={'names':('num','min_times','max_times','disruptive'),
                                                              'formats':('i4','f4','f4','i4')})
    shots = array(zip(*data)[0])
    min_times = array(zip(*data)[1])
    max_times = array(zip(*data)[2])
    disruptive = array(zip(*data)[3])
    return shots, min_times, max_times,disruptive

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


def get_multiple_shots_and_disruption_times(base_path,endings):
    all_shots = []
    all_disruption_times = []
    for ending in endings:
        path = base_path + ending
        shots,disruption_times = get_shots_and_disruption_times(path)
        all_shots.append(shots)
        all_disruption_times.append(disruption_times)
    return concatenate(all_shots),concatenate(all_disruption_times)








