from pylab import *
from matplotlib import pyplot
import os


def load_ith_file(i,mode='test',verbose=True):
    results_dir = '/p/datad/jkatesha/data/results/'
    results_files = os.listdir(results_dir)
    print(results_files)
    dat = load(results_dir + results_files[i])

    y_gold=dat['y_gold']
    y_gold_train=dat['y_gold_train']
    y_gold_test=dat['y_gold_test']
    y_prime=dat['y_prime']
    y_prime_train=dat['y_prime_train']
    y_prime_test=dat['y_prime_test']
    disruptive=dat['disruptive']
    disruptive_train=dat['disruptive_train']
    disruptive_test=dat['disruptive_test']

    if verbose:
        print('configuration: {} '.format(dat['conf']))

    if mode == 'test':
        pred = y_prime_test
        truth = y_gold_test
    elif mode == 'train':
        pred = y_prime_train
        truth = y_gold_train
    else:
        pred = y_prime
        truth = y_gold
    
    disruptive_curr = get_disruptive(truth) 
    print('loaded {} files ({}) disruptive in mode {}'.format(len(disruptive_curr),sum(disruptive_curr),mode))
   
    if any(disruptive_curr): 
        idx = where(disruptive_curr)[0][0]
        length = 1 + len(truth[idx]) - len(pred[idx])
    else:
	length = 1
    
    return pred,truth,disruptive_curr,length
    
    

def hist_alarms(alarms,T_min_warn,T_max_warn,title_str='alarms',save_figure=False):
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










def cut_ttd(arr,length):
    return arr[length-1:]


def get_disruptive(is_disr_list):
    return array([1 if any(arr > 0.5) else 0 for arr in is_disr_list])

  
def create_acceptable_regions(is_disrupt,acceptable_timesteps=1000):
    end_indices = get_end_indices(is_disrupt) 
    acceptable = zeros_like(is_disrupt,dtype=bool)
    for idx in end_indices:
        acceptable[idx - acceptable_timesteps:idx] = True
    return acceptable

def create_acceptable_region(is_disrupt,acceptable_timesteps=1000):
    acceptable = zeros_like(is_disrupt,dtype=bool)
    acceptable[-acceptable_timesteps:] = True
    return acceptable


def get_end_indices(is_disrupt):
    end_indices = where(logical_and(is_disrupt[:-1] > 0.5, is_disrupt[1:] < 0.5))[0]
    return end_indices

def get_accuracy_and_fp_rate(P_thresh,pred,is_disrupt,T_min_warn = 30,T_max_warn = 1000):
    predictions = pred > P_thresh
    predictions = reshape(predictions,(len(predictions),))
    
    max_acceptable = create_acceptable_region(is_disrupt,T_max_warn)
    min_acceptable = create_acceptable_region(is_disrupt,T_min_warn)
    
    tp = sum(logical_and(predictions,max_acceptable))
    fp = sum(logical_and(predictions,~max_acceptable))
    tn = sum(logical_and(~predictions,~min_acceptable))
    fn = sum(logical_and(~predictions,min_acceptable))
   
    # print(1.0*tp/(tp + fp))
    # print(1.0*tn/(tn + fn))
    # print(1.0*(tp + tn)/(tp + fp + tn + fn))
    print('total: {}, tp: {} fp: {} fn: {} tn: {}'.format(len(predictions),tp,fp,fn,tn))
    
   
    return get_accuracy_and_fp_rate_from_stats(tp,fp,fn)


def get_accuracy_and_fp_rate_from_stats(tp,fp,fn,tn,early,late,verbose=False):
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


#we are interested in the predictions of the *first alarm*
def get_shot_prediction_stats(P_thresh,pred,is_disrupt,is_disruptive,T_min_warn = 30,T_max_warn = 1000):
    predictions = pred > P_thresh
    predictions = reshape(predictions,(len(predictions),))
    
    
    max_acceptable = create_acceptable_region(is_disrupt,T_max_warn)
    min_acceptable = create_acceptable_region(is_disrupt,T_min_warn)
    
    early = 0
    late = 0
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    
    positives = where(predictions)[0]
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


def summarize_shot_prediction_stats(P_thresh,preds_by_shot,when_disrupt_by_shot,is_disruptive_by_shot,length,T_min_warn = 30,
                                   T_max_warn = 1000,verbose=False):
    TPs,FPs,FNs,TNs,earlies,lates = (0,0,0,0,0,0)
    when_disrupt_by_shot_cut = [cut_ttd(_ttd,length) for _ttd in when_disrupt_by_shot]
    for i in range(len(preds_by_shot)):
        preds = preds_by_shot[i]
        truth = when_disrupt_by_shot_cut[i]
        is_disruptive = is_disruptive_by_shot[i]


        TP,FP,FN,TN,early,late = get_shot_prediction_stats(P_thresh,preds,truth,is_disruptive,T_min_warn,T_max_warn)
        TPs += TP
        FPs += FP
        FNs += FN
        TNs += TN
        earlies += early
        lates += late
        
    # print(1.0*tp/(tp + fp))
    # print(1.0*tn/(tn + fn))
    # print(1.0*(tp + tn)/(tp + fp + tn + fn))
    disr = earlies + lates + TPs + FNs
    nondisr = FPs + TNs
    if verbose:
        print('total: {}, tp: {} fp: {} fn: {} tn: {} early: {} late: {} disr: {} nondisr: {}'.format(len(preds_by_shot),TPs,FPs,FNs,TNs,earlies,lates,disr,nondisr))
   
    return get_accuracy_and_fp_rate_from_stats(TPs,FPs,FNs,TNs,earlies,lates,verbose)

def gather_first_alarms(P_thresh,preds_by_shot,disruptive_by_shot,T_min_warn = 30,
                                   T_max_warn = 1000,verbose=False):
    
    alarms = []
    disr_alarms = []
    nondisr_alarms = []
    for i in range(len(preds_by_shot)):
        pred = preds_by_shot[i]
        predictions = pred > P_thresh
        predictions = reshape(predictions,(len(predictions),))
        positives = where(predictions)[0]
        if len(positives) > 0:
            alarm_ttd = len(pred) - 1.0 - positives[0]
            alarms.append(alarm_ttd)
            if disruptive_by_shot[i]:
                disr_alarms.append(alarm_ttd)
            else:
                nondisr_alarms.append(alarm_ttd)
        else:
            if disruptive_by_shot[i]:
                disr_alarms.append(-1)
    return array(alarms),array(disr_alarms),array(nondisr_alarms)
            
        
def get_metrics_vs_p_thresh(P_thresh_range,ttd_prime_by_shot,ttd_by_shot,disr,length, \
                           T_min_warn = 30,T_max_warn = 1000,verbose=False):
    correct_range = zeros_like(P_thresh_range)
    accuracy_range = zeros_like(P_thresh_range)
    fp_range = zeros_like(P_thresh_range)
    missed_range = zeros_like(P_thresh_range)
    early_alarm_range = zeros_like(P_thresh_range)

    verbose=False


#     ttd_cut_by_shot = [cut_ttd(_ttd,length) for _ttd in ttd_by_shot]
    # ttd_test_cut_by_shot = [cut_ttd(_ttd,length) for _ttd in ttd_by_shot]
    # ttd_train_cut_by_shot = [cut_ttd(_ttd,length) for _ttd in ttd_by_shot]

    for i,P_thresh in enumerate(P_thresh_range):
        correct,accuracy,fp_rate,missed,early_alarm_rate = summarize_shot_prediction_stats(P_thresh,ttd_prime_by_shot, \
                                ttd_by_shot,disr,length,T_min_warn,T_max_warn,verbose=verbose)
        correct_range[i] = correct
        accuracy_range[i] = accuracy 
        fp_range[i] = fp_rate 
        missed_range[i] = missed
        early_alarm_range[i] = early_alarm_rate
    
    return correct_range,accuracy_range,fp_range,missed_range,early_alarm_range


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


def compute_tradeoffs_and_print(P_thresh_range,pred,truth,disruptive_curr,length,T_min_warn,T_max_warn):
    correct_range, accuracy_range, fp_range,missed_range,early_alarm_range = get_metrics_vs_p_thresh(P_thresh_range, \
            pred,truth,disruptive_curr,length,T_min_warn,T_max_warn)

    fp_threshs = [0.01,0.05,0.1]
    missed_threshs = [0.01,0.05,0.0]


    #first index where...
    for fp_thresh in fp_threshs: 

        print('============= FP RATE < {} ============='.format(fp_thresh))
        if(any(fp_range < fp_thresh)):
            idx = where(fp_range <= fp_thresh)[0][0]
            P_thresh_opt = P_thresh_range[idx]
            summarize_shot_prediction_stats(P_thresh_opt,pred,truth,disruptive_curr,length,T_min_warn,T_max_warn,verbose=True)
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
            summarize_shot_prediction_stats(P_thresh_opt,pred,truth,disruptive_curr,length,T_min_warn,T_max_warn,verbose=True)
            print('============= AT P_THRESH = {} ============='.format(P_thresh_opt))
        else:
            print('No such P_thresh found')
        print('')


def compute_tradeoffs_and_print_from_training(P_thresh_range,pred_train,truth_train,disruptive_curr_train,pred_test,truth_test,disruptive_curr_test,length,T_min_warn,T_max_warn):
    correct_range, accuracy_range, fp_range,missed_range,early_alarm_range = get_metrics_vs_p_thresh(P_thresh_range, \
            pred_train,truth_train,disruptive_curr_train,length,T_min_warn,T_max_warn)

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
            summarize_shot_prediction_stats(P_thresh_opt,pred_test,truth_test,disruptive_curr_test,length,T_min_warn,T_max_warn,verbose=True)
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
            summarize_shot_prediction_stats(P_thresh_opt,pred_test,truth_test,disruptive_curr_test,length,T_min_warn,T_max_warn,verbose=True)
	    if missed_thresh == 0.05:
		P_thresh_ret = P_thresh_opt
            print('============= AT P_THRESH = {} ============='.format(P_thresh_opt))
        else:
            print('No such P_thresh found')
	    P_thresh_opt = P_thresh_default
        print('')
    return P_thresh_ret




def tradeoff_plot(P_thresh_range,accuracy_range,missed_range,fp_range,early_alarm_range,save_figure=False):
    figure()
    semilogx(P_thresh_range,accuracy_range,label="accuracy")
    plot(P_thresh_range,missed_range,'r',label="missed")
    plot(P_thresh_range,fp_range,'k',label="false positives")
    plot(P_thresh_range,early_alarm_range,'c',label="early alarms")
    legend(loc=(1.0,.6))
    xlabel('Alarm threshold')
    grid()
    if save_figure:
        savefig('metrics.png',bbox_inches='tight')



def compute_tradeoffs_and_plot(P_thresh_range,pred,truth,disruptive_curr,length,T_min_warn,T_max_warn,save_figure=True):
    correct_range, accuracy_range, fp_range,missed_range,early_alarm_range = get_metrics_vs_p_thresh(P_thresh_range, \
            pred,truth,disruptive_curr,length,T_min_warn,T_max_warn)

    tradeoff_plot(P_thresh_range,accuracy_range,missed_range,fp_range,early_alarm_range,save_figure=save_figure)



