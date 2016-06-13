import matplotlib
matplotlib.use('Agg')

from pylab import *
from matplotlib import pyplot
import os

from performance_analysis_utils import *

mode = 'test'
file_num = 0
save_figure = True
pred_ttd = False

if pred_ttd:
    P_thresh_range = -1*logspace(0,-4,100)#concatenate((linspace(-1,-0.10,5),linspace(-0.1,0,100)))#array([-0.2,-0.1,-0.05,-0.02,-0.01,-0.005,-0.002,-0.001,0])#  #linspace(-2,0,50)
else:
    P_thresh_range = logspace(-4,0,100) 

T_max_warn = 1000
T_min_warn = 30

verbose=False
results_dir = '/tigress/jk7/data/results/'
shots_dir = '/tigress/jk7/data/processed_shots/'#'/p/datad/jkatesha/data/results/'

analyzer = PerformanceAnalyzer(results_dir=results_dir,shots_dir=shots_dir,i = file_num,
T_min_warn = T_min_warn,T_max_warn = T_max_warn, verbose = verbose, pred_ttd=pred_ttd) 

analyzer.load_ith_file()

#compute_tradeoffs_and_print(P_thresh_range,pred,truth,disruptive_curr,length,T_min_warn,T_max_warn)

P_thresh_opt = analyzer.compute_tradeoffs_and_print_from_training(P_thresh_range)

analyzer.compute_tradeoffs_and_plot(P_thresh_range,'train',save_figure=save_figure,plot_string='_train')
analyzer.compute_tradeoffs_and_plot(P_thresh_range,'test',save_figure=save_figure,plot_string='_test')

analyzer.summarize_shot_prediction_stats(P_thresh_opt,'test')

#analyzer.example_plots(P_thresh_opt,'test','any')
analyzer.example_plots(P_thresh_opt,'test','FP')
analyzer.example_plots(P_thresh_opt,'test','FN')
analyzer.example_plots(P_thresh_opt,'test','TP')
analyzer.example_plots(P_thresh_opt,'test','late')


alarms,disr_alarms,nondisr_alarms = analyzer.gather_first_alarms(P_thresh_opt,'test')
analyzer.hist_alarms(disr_alarms,'disruptive alarms, P_thresh = {}'.format(P_thresh_opt),save_figure=save_figure)
print('{} disruptive alarms'.format(len(disr_alarms)))
print('{} seconds mean alarm time'.format(mean(disr_alarms[disr_alarms > 0])))
analyzer.hist_alarms(nondisr_alarms,'nondisruptive alarms, P_thresh = {}'.format(P_thresh_opt))
print('{} nondisruptive alarms'.format(len(nondisr_alarms)))

