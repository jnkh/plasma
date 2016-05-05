from pylab import *
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.utils.data_utils import get_file
from scipy.cluster.vq import whiten
import numpy as np
import random
import sys



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


def get_signals_and_ttds(signal_prepath,signals_dirs,shots,min_times,max_times,T_max,dt,use_shots=3):
    all_signals = []
    all_ttd = []
    use_shots = min([use_shots,len(shots)-1])
    for (j,shot_num) in enumerate(shots[:use_shots]):
        shot = shots[j]
        t_min = min_times[j]
        t_max = max_times[j]
        signals,ttd = get_signal_and_ttd(signal_prepath,signals_dirs,shot,t_min,t_max,T_max,dt)
        all_signals.append(signals)
        all_ttd.append(ttd)
        print(1.0*j/use_shots)

    signals = vstack(all_signals)
    ttd = hstack(all_ttd)
    return signals,ttd

def get_signal_and_ttd(signal_prepath,signals_dirs,shot,t_min,t_max,T_max,dt):
    signals = []
    times = []
    for (i,dirname) in enumerate(signals_dirs):
        data = loadtxt(signal_prepath+dirname + '/' + str(shot) + '.txt')
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
    data = loadtxt(shots_and_times_path,dtype={'names':('num','timemin','timemax'),
                                                              'formats':('i4','f4','f4')})
    shots = array(zip(*data)[0])
    min_times = array(zip(*data)[1])
    max_times = array(zip(*data)[2])
    return shots,min_times,max_times
