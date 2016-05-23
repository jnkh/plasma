from matplotlib import rc
rc('text', usetex=True)

import matplotlib
matplotlib.use('Agg')

from pylab import *
from matplotlib import pyplot
import os

from performance_analysis_utils import *

mode = 'test'
file_num = 2
save_figure = False

P_thresh_range = logspace(-4,0,100) 
T_max_warn = 1000
T_min_warn = 30

verbose=False
results_dir = '../data/results/'


print("preprocessing all shots")
pp = Preprocessor(conf)
pp.clean_shot_lists()
shot_list = pp.preprocess_all()
sorted(shot_list)
shot_list_train,shot_list_test = shot_list.split_train_test(conf)
num_shots = len(shot_list_train) + len(shot_list_test)
print("...done")

nn = Normalizer(conf)
nn.train()
loader = Loader(conf,nn)


def plot_shot_from_file(j,conf,shots,disruption_times,processed_prepath,standard_deviations,labels,whiten=whiten):
    # num_processed += 1
    # print('({}/{}): '.format(num_processed,use_shots))
    shot = shots[j]
    t_disrupt = disruption_times[j]
    load_file_path = get_individual_shot_file(processed_prepath,shot,'.npz')
    is_disruptive =  t_disrupt >= 0

    signals,times,t_min,t_max,t_thresh,valid = get_signals_and_times_from_file(shot,t_disrupt,conf) 
    signals,ttd = cut_and_resample_signals(times,signals,t_min,t_max,is_disruptive,conf,standard_deviations,whiten=whiten)
    print('shot {}'.format(shot))
    if is_disruptive:
    	print('disruptive')
    else:
    	print('non disruptive')

    f,axarr = subplots(len(signals.T)/2,2)
    for (i,sig) in enumerate(signals.T):
    	axarr.flatten()[i].plot(sig,label = labels[i])
    	axarr.flatten()[i].legend(loc='best')
    	print('min: {}, max: {}'.format(min(sig), max(sig)))
    show() 



labels = [r' $I_{plasma}$ [A]',
r' Mode L. A. [A]',
r' $P_{radiated}$ [W]',
r' $\rho_{plasma}$ [m^-2]',
r' $L_{plasma,internal}$',
r'$\frac{d}{dt} E_{D}$ [W]',
r' $P_{input}$ [W]',
r'$E_{D}$']



whiten = False

shot_list_dir = conf['paths']['shot_list_dir']
shot_files = conf['paths']['shot_files']
shot_files_test = conf['paths']['shot_files_test']
train_frac = conf['training']['train_frac']


shots,disruption_times = get_multiple_shots_and_disruption_times(shot_list_dir,shot_files)

standard_deviations = preprocess_data_whitener(conf)

dt = conf['data']['dt']
processed_prepath = conf['paths']['processed_prepath']
recompute = conf['data']['recompute']
use_shots = 1 
used_shots = []
disruptive = []
indices = np.random.choice(arange(len(shots)),size=use_shots,replace=False)


plot_shot_from_file(indices[0],conf,shots,disruption_times,processed_prepath,standard_deviations,labels,whiten=False)


