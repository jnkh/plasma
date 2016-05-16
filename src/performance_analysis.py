import matplotlib
matplotlib.use('Agg')

from pylab import *
from matplotlib import pyplot
import os

from performance_analysis_utils import *

mode = 'test'
file_num = -1
save_figure = True

P_thresh_range = logspace(-4,0,100) 
T_max_warn = 1000
T_min_warn = 30
P_thresh_opt = 0.04

pred,truth,disruptive_curr,length = load_ith_file(file_num,mode)

pred_train,truth_train,disruptive_curr_train,length = load_ith_file(file_num,'train')

#compute_tradeoffs_and_print(P_thresh_range,pred,truth,disruptive_curr,length,T_min_warn,T_max_warn)

compute_tradeoffs_and_print_from_training(P_thresh_range,pred_train,truth_train,disruptive_curr_train,pred,truth,disruptive_curr,length,T_min_warn,T_max_warn)


compute_tradeoffs_and_plot(P_thresh_range,pred,truth,disruptive_curr,length,T_min_warn,T_max_warn,save_figure=save_figure)

summarize_shot_prediction_stats(P_thresh_opt,pred,truth,disruptive_curr,length,T_min_warn,T_max_warn,verbose=True)



alarms,disr_alarms,nondisr_alarms = gather_first_alarms(P_thresh_opt,pred,disruptive_curr,T_min_warn,T_max_warn)
hist_alarms(disr_alarms,T_min_warn,T_max_warn,'disruptive alarms',save_figure=save_figure)
print('{} disruptive alarms'.format(len(disr_alarms)))
print('{} seconds mean alarm time'.format(mean(disr_alarms[disr_alarms > 0])))
hist_alarms(nondisr_alarms,T_min_warn,T_max_warn,'nondisruptive alarms')
print('{} nondisruptive alarms'.format(len(nondisr_alarms)))

