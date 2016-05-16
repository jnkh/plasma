from pylab import *
from matplotlib import pyplot
import os
%load_ext autoreload
%autoreload 2

from performance_analysis_utils import *

mode = 'test'
file_num = 1
save_figure = True

P_thresh_range = logspace(-6,0,50) 
T_max_warn = 1000
T_min_warn = 30
P_thresh_opt = 0.002

pred,truth,disruptive_curr,length = load_ith_file(file_num,mode)

compute_tradeoffs_and_plot(P_thresh_range,pred,truth,disruptive_curr,length,T_min_warn,T_max_warn,save_figure=save_figure)

summarize_shot_prediction_stats(P_thresh_opt,pred,truth,disruptive_curr,length,T_min_warn,T_max_warn,verbose=True)
