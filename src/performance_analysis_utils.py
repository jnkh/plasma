from pylab import *
from matplotlib import pyplot
import os



class PerformanceAnalyzer():
    def __init__(self,results_dir=None,i = 0,T_min_warn = 30,T_max_warn = 1000, verbose = False):
        self.T_min_warn = T_min_warn
        self.T_max_warn = T_max_warn
        self.verbose = verbose
        self.results_dir = results_dir
        self.i = i

        self.pred_train = None
        self.truth_train = None
        self.disruptive_train = None

        self.pred_test = None
        self.truth_test = None
        self.disruptive_test = None



    def get_metrics_vs_p_thresh(self,P_thresh_range,mode):
        correct_range = zeros_like(P_thresh_range)
        accuracy_range = zeros_like(P_thresh_range)
        fp_range = zeros_like(P_thresh_range)
        missed_range = zeros_like(P_thresh_range)
        early_alarm_range = zeros_like(P_thresh_range)

        for i,P_thresh in enumerate(P_thresh_range):
            correct,accuracy,fp_rate,missed,early_alarm_rate = self.summarize_shot_prediction_stats(P_thresh,mode=mode)
            correct_range[i] = correct
            accuracy_range[i] = accuracy 
            fp_range[i] = fp_rate 
            missed_range[i] = missed
            early_alarm_range[i] = early_alarm_rate
        
        return correct_range,accuracy_range,fp_range,missed_range,early_alarm_range


    def summarize_shot_prediction_stats(self,P_thresh,mode,verbose=False):
        if mode == 'train':
            all_preds = self.pred_train
            all_truths = self.truth_train
            all_disruptive = self.disruptive_train


        elif mode == 'test':
            all_preds = self.pred_test
            all_truths = self.truth_test
            all_disruptive = self.disruptive_test

        TPs,FPs,FNs,TNs,earlies,lates = (0,0,0,0,0,0)

        for i in range(len(all_preds)):
            preds = all_preds[i]
            truth = all_truths[i]
            is_disruptive = all_disruptive[i]


            TP,FP,FN,TN,early,late = self.get_shot_prediction_stats(P_thresh,preds,truth,is_disruptive)
            TPs += TP
            FPs += FP
            FNs += FN
            TNs += TN
            earlies += early
            lates += late
            
        disr = earlies + lates + TPs + FNs
        nondisr = FPs + TNs
        if verbose:
            print('total: {}, tp: {} fp: {} fn: {} tn: {} early: {} late: {} disr: {} nondisr: {}'.format(len(all_preds),TPs,FPs,FNs,TNs,earlies,lates,disr,nondisr))
       
        return self.get_accuracy_and_fp_rate_from_stats(TPs,FPs,FNs,TNs,earlies,lates,verbose)

    #we are interested in the predictions of the *first alarm*
    def get_shot_prediction_stats(self,P_thresh,pred,truth,is_disruptive):
        predictions = pred > P_thresh
        predictions = reshape(predictions,(len(predictions),))
        
        max_acceptable = self.create_acceptable_region(truth,'max')
        min_acceptable = self.create_acceptable_region(truth,'min')
        
        early = late = TP = TN = FN = FP = 0
      
        positives = self.get_positives(predictions)#where(predictions)[0]
        if len(positives) == 0:
            if is_disruptive:
                FN = 1
            else:
                TN = 1
        else:
            if is_disruptive:
                first_pred_idx = positives[0]
                if max_acceptable[first_pred_idx] and ~min_acceptable[first_pred_idx]:
                    TP = 1
                elif min_acceptable[first_pred_idx]:
                    late = 1
                elif ~max_acceptable[first_pred_idx]:
                    early = 1
            else:
                FP = 1
        return TP,FP,FN,TN,early,late


    def get_positives(self,predictions):
        indices = arange(len(predictions))
        return where(logical_and(predictions,indices >= 100))[0]


    def create_acceptable_region(self,truth,mode):
        if mode == 'min':
            acceptable_timesteps = self.T_min_warn
        elif mode == 'max':
            acceptable_timesteps = self.T_max_warn
        else:
            print('Error Invalid Mode for acceptable region')
            exit(1) 

        acceptable = zeros_like(truth,dtype=bool)
        acceptable[-acceptable_timesteps:] = True
        return acceptable


    def get_accuracy_and_fp_rate_from_stats(self,tp,fp,fn,tn,early,late,verbose=False):
        total = tp + fp + fn + tn + early + late
        disr = early + late + tp + fn 
        nondisr = fp + tn
        
        if disr == 0:
            early_alarm_rate = 0
            missed = 0
            accuracy = 0 
        else:
            early_alarm_rate = 1.0*early/disr
            missed = 1.0*(late + fn)/disr
            accuracy = 1.0*tp/disr
        if nondisr == 0:
            fp_rate = 0
        else: 
            fp_rate = 1.0*fp/nondisr
        correct = 1.0*(tp + tn)/total
        
        if verbose:
            print('accuracy: {}'.format(accuracy))
            print('missed: {}'.format(missed))
            print('early alarms: {}'.format(early_alarm_rate))
            print('false positive rate: {}'.format(fp_rate))
            print('correct: {}'.format(correct))

        return correct,accuracy,fp_rate,missed,early_alarm_rate



    def load_ith_file(self):
        results_files = os.listdir(self.results_dir)
        print(results_files)
        dat = load(self.results_dir + results_files[self.i])

        if self.verbose:
            print('configuration: {} '.format(dat['conf']))

        self.pred_train = dat['y_prime_train']
        self.truth_train = dat['y_gold_train']
        self.disruptive_train = dat['disruptive_train']
        self.pred_test = dat['y_prime_test']
        self.truth_test = dat['y_gold_test']
        self.disruptive_test = dat['disruptive_test']
        for mode in ['test','train']:
            print('{}: loaded {} shot ({}) disruptive'.format(mode,self.get_num_shots(mode),self.get_num_disruptive_shots(mode)))
    

    def get_num_shots(self,mode):
        if mode == 'test':
            return len(self.disruptive_test)
        if mode == 'train':
            return len(self.disruptive_train)

    def get_num_disruptive_shots(self,mode):
        if mode == 'test':
            return sum(self.disruptive_test)
        if mode == 'train':
            return sum(self.disruptive_train)


    def hist_alarms(self,alarms,title_str='alarms',save_figure=False):
        T_min_warn = self.T_min_warn
        T_max_warn = self.T_max_warn
        if len(alarms) > 0:
            alarms /= 1000.0
            alarms = sort(alarms)
            T_min_warn /= 1000.0
            T_max_warn /= 1000.0
            figure()
            alarms += 0.0001
            bins=logspace(log10(min(alarms)),log10(max(alarms)),40)
            #bins=linspace(min(alarms),max(alarms),100)
            #        hist(alarms,bins=bins,alpha=1.0,histtype='step',normed=True,log=False,cumulative=-1)
            #
            pyplot.step(np.concatenate((alarms[::-1], alarms[[0]])), 1.0*np.arange(alarms.size+1)/(alarms.size))

            gca().set_xscale('log')
            axvline(T_min_warn,color='r')
            axvline(T_max_warn,color='r')
            xlabel('TTD [s]')
            ylabel('Accumulated fraction of detected disruptions')
            xlim([1e-4,max(alarms)*10])
            ylim([0,1])
            grid()
            title(title_str)
            show()
    	if save_figure:
    	    savefig('accum_disruptions.png',bbox_inches='tight')
        else:
            print(title_str + ": No alarms!")



    def gather_first_alarms(self,P_thresh,mode):
        if mode == 'train':
            pred_list = self.pred_train 
            disruptive_list = self.disruptive_train 
        elif mode == 'test':
            pred_list = self.pred_test 
            disruptive_list = self.disruptive_test 
        
        alarms = []
        disr_alarms = []
        nondisr_alarms = []
        for i in range(len(pred_list)):
            pred = pred_list[i]
            predictions = pred > P_thresh
            predictions = reshape(predictions,(len(predictions),))
            positives = self.get_positives(predictions)#where(predictions)[0]
            if len(positives) > 0:
                alarm_ttd = len(pred) - 1.0 - positives[0]
                alarms.append(alarm_ttd)
                if disruptive_list[i]:
                    disr_alarms.append(alarm_ttd)
                else:
                    nondisr_alarms.append(alarm_ttd)
            else:
                if disruptive_list[i]:
                    disr_alarms.append(-1)
        return array(alarms),array(disr_alarms),array(nondisr_alarms)
                
            



    def compute_tradeoffs_and_print(self,P_thresh_range,mode):
        correct_range, accuracy_range, fp_range,missed_range,early_alarm_range = self.get_metrics_vs_p_thresh(P_thresh_range,mode)
        fp_threshs = [0.01,0.05,0.1]
        missed_threshs = [0.01,0.05,0.0]

        #first index where...
        for fp_thresh in fp_threshs: 
            print('============= FP RATE < {} ============='.format(fp_thresh))
            if(any(fp_range < fp_thresh)):
                idx = where(fp_range <= fp_thresh)[0][0]
                P_thresh_opt = P_thresh_range[idx]
                self.summarize_shot_prediction_stats(P_thresh_opt,mode,verbose=True)
                print('============= AT P_THRESH = {} ============='.format(P_thresh_opt))
            else:
                print('No such P_thresh found')
            print('')

        #last index where
        for missed_thresh in missed_threshs: 
            print('============= MISSED RATE < {} ============='.format(missed_thresh))
            if(any(missed_range < missed_thresh)):
                idx = where(missed_range <= missed_thresh)[0][-1]
                P_thresh_opt = P_thresh_range[idx]
                self.summarize_shot_prediction_stats(P_thresh_opt,mode,verbose=True)
                print('============= AT P_THRESH = {} ============='.format(P_thresh_opt))
            else:
                print('No such P_thresh found')
            print('')

        print('============== Crossing Point: ==============')
        print('============= TEST PERFORMANCE: =============')
        idx = where(missed_range <= fp_range)[0][-1]
        P_thresh_opt = P_thresh_range[idx]
        self.summarize_shot_prediction_stats(P_thresh_opt,mode,verbose=True)
        P_thresh_ret = P_thresh_opt
        return P_thresh_ret


    def compute_tradeoffs_and_print_from_training(self,P_thresh_range):
        correct_range, accuracy_range, fp_range,missed_range,early_alarm_range = self.get_metrics_vs_p_thresh(P_thresh_range,'train')

        fp_threshs = [0.01,0.05,0.1]
        missed_threshs = [0.01,0.05,0.0]
        P_thresh_default = 0.03
        P_thresh_ret = P_thresh_default

        #first index where...
        for fp_thresh in fp_threshs: 

            print('============= TRAINING FP RATE < {} ============='.format(fp_thresh))
            print('============= TEST PERFORMANCE: =============')
            if(any(fp_range < fp_thresh)):
                idx = where(fp_range <= fp_thresh)[0][0]
                P_thresh_opt = P_thresh_range[idx]
                self.summarize_shot_prediction_stats(P_thresh_opt,'test',verbose=True)
                print('============= AT P_THRESH = {} ============='.format(P_thresh_opt))
            else:
                print('No such P_thresh found')
    	    P_thresh_opt = P_thresh_default
            print('')

        #last index where
        for missed_thresh in missed_threshs: 

            print('============= TRAINING MISSED RATE < {} ============='.format(missed_thresh))
            print('============= TEST PERFORMANCE: =============')
            if(any(missed_range < missed_thresh)):
                idx = where(missed_range <= missed_thresh)[0][-1]
                P_thresh_opt = P_thresh_range[idx]
                self.summarize_shot_prediction_stats(P_thresh_opt,'test',verbose=True)
                if missed_thresh == 0.05:
                    P_thresh_ret = P_thresh_opt
                print('============= AT P_THRESH = {} ============='.format(P_thresh_opt))
            else:
                print('No such P_thresh found')
    	    P_thresh_opt = P_thresh_default
            print('')

        print('============== Crossing Point: ==============')
        print('============= TEST PERFORMANCE: =============')
        idx = where(missed_range <= fp_range)[0][-1]
        P_thresh_opt = P_thresh_range[idx]
        self.summarize_shot_prediction_stats(P_thresh_opt,'test',verbose=True)
        P_thresh_ret = P_thresh_opt
        return P_thresh_ret


    def compute_tradeoffs_and_plot(self,P_thresh_range,mode,save_figure=True,plot_string=''):
        correct_range, accuracy_range, fp_range,missed_range,early_alarm_range = self.get_metrics_vs_p_thresh(P_thresh_range,mode)

        tradeoff_plot(P_thresh_range,accuracy_range,missed_range,fp_range,early_alarm_range,save_figure=save_figure,plot_string=plot_string)

    def example_plots(self,P_thresh_opt,mode='test',type = 'FP'):
        if mode == 'test':
            pred = self.pred_test
            truth = self.truth_test
            is_disruptive = self.disruptive_test
        else:
            pred = self.pred_train
            truth = self.truth_train
            is_disruptive = self.disruptive_train
        to_plot = 5
        max_plot = 5
        plotted = 0
        iterate_arr = range(len(truth))
        shuffle(iterate_arr)
        for i in iterate_arr:
            t = truth[i]
            p = pred[i]
            is_disr = is_disruptive[i]
            TP,FP,FN,TN,early,late =self.get_shot_prediction_stats(P_thresh_opt,p,t,is_disr)
            if type == 'FP':
                comparison = FP
            elif type == 'TP':
                comparison =TP 
            elif type == 'FN':
                comparison =FN 
            elif type == 'TN':
                comparison =TN 
            elif type == 'late':
                comparison =late 
            elif type == 'early':
                comparison =early
            elif type == 'any':
                comparison = True
            else:
                print('warning, unkown type')
                return
            if comparison and plotted < max_plot:
                figure()
                loglog((t+0.001)[::-1],label='ground truth')
                plot(p[::-1],'g',label='neural net prediction')
                axvline(self.T_min_warn,color='r',label='max warning time')
                axvline(self.T_max_warn,color='r',label='min warning time')
                axhline(P_thresh_opt,color='k',label='trigger threshold')
                xlabel('TTD [ms]')
                legend(loc = (1.0,0.6))
                ylim([1e-7,1e0])
                grid()
                plotted += 1

def tradeoff_plot(P_thresh_range,accuracy_range,missed_range,fp_range,early_alarm_range,save_figure=False,plot_string=''):
    figure()
    semilogx(P_thresh_range,accuracy_range,label="accuracy")
    plot(P_thresh_range,missed_range,'r',label="missed")
    plot(P_thresh_range,fp_range,'k',label="false positives")
    plot(P_thresh_range,early_alarm_range,'c',label="early alarms")
    legend(loc=(1.0,.6))
    xlabel('Alarm threshold')
    grid()
    title_str = 'metrics{}'.format(plot_string)
    title(title_str)
    if save_figure:
        savefig(title_str + '.png',bbox_inches='tight')




# def cut_ttd(arr,length):
#     return arr[length-1:]


# def get_disruptive(is_disr_list):
#     return array([1 if any(arr > 0.5) else 0 for arr in is_disr_list])

  
# def create_acceptable_regions(is_disrupt):
#     end_indices = get_end_indices(is_disrupt) 
#     acceptable = zeros_like(is_disrupt,dtype=bool)
#     for idx in end_indices:
#         acceptable[idx - acceptable_timesteps:idx] = True
#     return acceptable

# def get_end_indices(is_disrupt):
#     end_indices = where(logical_and(is_disrupt[:-1] > 0.5, is_disrupt[1:] < 0.5))[0]
#     return end_indices

# def get_accuracy_and_fp_rate(P_thresh,pred,is_disrupt,T_min_warn = 30,T_max_warn = 1000):
#     predictions = pred > P_thresh
#     predictions = reshape(predictions,(len(predictions),))
    
#     max_acceptable = create_acceptable_region(is_disrupt,T_max_warn)
#     min_acceptable = create_acceptable_region(is_disrupt,T_min_warn)
    
#     tp = sum(logical_and(predictions,max_acceptable))
#     fp = sum(logical_and(predictions,~max_acceptable))
#     tn = sum(logical_and(~predictions,~min_acceptable))
#     fn = sum(logical_and(~predictions,min_acceptable))
   
#     # print(1.0*tp/(tp + fp))
#     # print(1.0*tn/(tn + fn))
#     # print(1.0*(tp + tn)/(tp + fp + tn + fn))
#     print('total: {}, tp: {} fp: {} fn: {} tn: {}'.format(len(predictions),tp,fp,fn,tn))
    
   
#     return get_accuracy_and_fp_rate_from_stats(tp,fp,fn)


# def get_thresholds(ttd_prime_by_shot,ttd_by_shot,disr,length, \
#                            T_min_warn = 30,T_max_warn = 1000,verbose=False):
    
#     def fp_vs_thresh(P_thresh):
#         correct,accuracy,fp_rate,missed,early_alarm_rate = summarize_shot_prediction_stats(P_thresh,ttd_prime_by_shot, \
#                                 ttd_by_shot,disr,length,T_min_warn,T_max_warn,verbose=verbose)
#         return fp_rate

#     def missed_vs_thresh(P_thresh):
#         correct,accuracy,fp_rate,missed,early_alarm_rate = summarize_shot_prediction_stats(P_thresh,ttd_prime_by_shot, \
#                                 ttd_by_shot,disr,length,T_min_warn,T_max_warn,verbose=verbose)
#         return fp_rate

