import matplotlib
matplotlib.use('Agg')

from pylab import *
from matplotlib import pyplot
import os

from performance_analysis_utils import *

mode = 'test'
file_num = 0
save_figure = True

P_thresh_range = logspace(-4,0,100) 
T_max_warn = 1000
T_min_warn = 30

verbose=False
results_dir = '/p/datad/jkatesha/data/results/'


analyzer = PerformanceAnalyzer(results_dir=results_dir,i = file_num,
	T_min_warn = T_min_warn,T_max_warn = T_max_warn, verbose = verbose) 

analyzer.load_ith_file()

#compute_tradeoffs_and_print(P_thresh_range,pred,truth,disruptive_curr,length,T_min_warn,T_max_warn)

P_thresh_opt = analyzer.compute_tradeoffs_and_print_from_training(P_thresh_range)

analyzer.compute_tradeoffs_and_plot(P_thresh_range,'train',save_figure=save_figure,plot_string='_train')
analyzer.compute_tradeoffs_and_plot(P_thresh_range,'test',save_figure=save_figure,plot_string='_test')

analyzer.summarize_shot_prediction_stats(P_thresh_opt,'test')

alarms,disr_alarms,nondisr_alarms = analyzer.gather_first_alarms(P_thresh_opt,'test')
analyzer.hist_alarms(disr_alarms,'disruptive alarms, P_thresh = {}'.format(P_thresh_opt),save_figure=save_figure)
print('{} disruptive alarms'.format(len(disr_alarms)))
print('{} seconds mean alarm time'.format(mean(disr_alarms[disr_alarms > 0])))
analyzer.hist_alarms(nondisr_alarms,'nondisruptive alarms, P_thresh = {}'.format(P_thresh_opt))
print('{} nondisruptive alarms'.format(len(nondisr_alarms)))

