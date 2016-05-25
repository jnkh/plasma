'''
#########################################################
This file containts classes to handle data processing

Author: Julian Kates-Harbeck, jkatesharbeck@g.harvard.edu

This work was supported by the DOE CSGF program.
#########################################################
'''

from __future__ import print_function
from os import listdir,remove
import os.path
import time,random,sys

from pylab import *
import numpy as np
from scipy.interpolate import UnivariateSpline

import pathos.multiprocessing as mp



#######NORMALIZATION##########

class Stats(object):
    pass

class Normalizer(object):
    def __init__(self,conf):
        self.minimums = None
        self.maximums = None
        self.num_processed = 0
        self.num_disruptive = 0
        self.conf = conf
        self.path = conf['paths']['normalizer_path']
        self.remapper = conf['data']['ttd_remapper']


    def __str__(self):
        return('Normalizer.\nminimums: {}\nmaximums: {}'.format(self.minimums,self.maximums))

    def extract_stats(self,shot):
        stats = Stats()
        if shot.valid:
            stats.minimums = np.min(shot.signals,0)
            stats.maximums = np.max(shot.signals,0)
            stats.is_disruptive = shot.is_disruptive
        else:
            print('Warning: shot {} not valid, omitting'.format(shot))
        stats.valid = shot.valid
        return stats


    def incorporate_stats(self,stats):
        if stats.valid:
            minimums = stats.minimums
            maximums = stats.maximums
            if self.num_processed == 0:
                self.minimums = minimums
                self.maximums = maximums
            else:
                self.minimums = (self.num_processed*self.minimums + minimums)/(self.num_processed + 1.0)#snp.min(vstack((self.minimums,minimums)),0)
                self.maximums = (self.num_processed*self.maximums + maximums)/(self.num_processed + 1.0)#snp.max(vstack((self.maximums,maximums)),0)
            self.num_processed = self.num_processed + 1
            self.num_disruptive = self.num_disruptive + (1 if stats.is_disruptive else 0)


    def apply(self,shot):
        assert(self.minimums is not None and self.maximums is not None) 
        shot.signals = (shot.signals - self.minimums)/(self.maximums - self.minimums)
        shot.ttd = self.remapper(shot.ttd,self.conf['data']['T_warning'])

    def save_stats(self):
        # standard_deviations = dat['standard_deviations']
        # num_processed = dat['num_processed']
        # num_disruptive = dat['num_disruptive']
        np.savez(self.path,minimums = self.minimums,maximums = self.maximums,
         num_processed=self.num_processed,num_disruptive=self.num_disruptive)
        print('saved normalization data from {} shots ( {} disruptive )'.format(self.num_processed,self.num_disruptive))

    def load_stats(self):
        assert(self.previously_saved_stats())
        dat = load(self.path)
        self.minimums = dat['minimums']
        self.maximums = dat['maximums']
        self.num_processed = dat['num_processed']
        self.num_disruptive = dat['num_disruptive']
        print('loaded normalization data from {} shots ( {} disruptive )'.format(self.num_processed,self.num_disruptive))
        #print('loading normalization data from {} shots, {} disruptive'.format(num_processed,num_disruptive))

    ######Modify the above to change the specifics of the normalization scheme#######

    def train(self):
        conf = self.conf
        #only use training shots here!! "Don't touch testing shots"
        shot_files = conf['paths']['shot_files']# + conf['paths']['shot_files_test']
        shot_list_dir = conf['paths']['shot_list_dir']
        use_shots = max(100,int(round(0.1*conf['data']['use_shots'])))
        return self.train_on_files(shot_list_dir,shot_files,use_shots)


    def train_on_files(self,shot_list_dir,shot_files,use_shots):
        conf = self.conf
        shot_list = ShotList()
        shot_list.load_from_files(shot_list_dir,shot_files)

        recompute = conf['data']['recompute_normalization']
        shot_list_picked = shot_list.random_sublist(use_shots) 

        if recompute or not self.previously_saved_stats():
            pool = mp.Pool()
            print('running in parallel on {} processes'.format(pool._processes))
            start_time = time.time()

            for (i,stats) in enumerate(pool.imap_unordered(self.train_on_single_shot,shot_list_picked)):
                self.incorporate_stats(stats)
                print('{}/{}'.format(i,len(shot_list_picked)))

            pool.close()
            pool.join()
            print('Finished Training Normalizer on {} files in {} seconds'.format(len(shot_list_picked),time.time()-start_time))
            self.save_stats()
        else:
            self.load_stats()
        print(self)


    def train_on_single_shot(self,shot):
        assert(isinstance(shot,Shot))
        processed_prepath = self.conf['paths']['processed_prepath']
        shot.restore(processed_prepath)
        stats = self.extract_stats(shot) 
        shot.make_light()
        return stats


    def previously_saved_stats(self):
        return os.path.isfile(self.path)



class Preprocessor(object):

    def __init__(self,conf):
        self.conf = conf


    def clean_shot_lists(self):
        shot_list_dir = self.conf['paths']['shot_list_dir']
        paths = [os.path.join(shot_list_dir, f) for f in listdir(shot_list_dir) if os.path.isfile(os.path.join(shot_list_dir, f))]
        for path in paths:
            self.clean_shot_list(path)


    def clean_shot_list(self,path):
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


    def preprocess_all(self):
        conf = self.conf
        shot_files = conf['paths']['shot_files'] + conf['paths']['shot_files_test']
        shot_list_dir = conf['paths']['shot_list_dir']
        use_shots = conf['data']['use_shots']
        return self.preprocess_from_files(shot_list_dir,shot_files,use_shots)


    def preprocess_from_files(self,shot_list_dir,shot_files,use_shots):
        #all shots, including invalid ones
        shot_list = ShotList()
        shot_list.load_from_files(shot_list_dir,shot_files)

        shot_list_picked = shot_list.random_sublist(use_shots)
        #empty
        used_shots = ShotList()

        pool = mp.Pool()

        print('running in parallel on {} processes'.format(pool._processes))
        start_time = time.time()
        for (i,shot) in enumerate(pool.imap_unordered(self.preprocess_single_file,shot_list_picked)):
        # for (i,shot) in enumerate(pool.imap_unordered(self.preprocess_single_file,[1,2,3])):
            print('{}/{}'.format(i,len(shot_list_picked)))
            used_shots.append_if_valid(shot)

        pool.close()
        pool.join()
        print('Finished Preprocessing {} files in {} seconds'.format(len(shot_list_picked),time.time()-start_time))
        print('Omitted {} shots of {} total.'.format(len(shot_list_picked) - len(used_shots),len(shot_list_picked)))
        print('{}/{} disruptive shots'.format(used_shots.num_disruptive(),len(used_shots)))
        return used_shots 

    def preprocess_single_file(self,shot):
        processed_prepath = self.conf['paths']['processed_prepath']
        recompute = self.conf['data']['recompute']
        # print('({}/{}): '.format(num_processed,use_shots))
        if recompute or not shot.previously_saved(processed_prepath):
            # print('(re)computing shot_number {}'.format(shot_number),end='')
          #get minmax times
            signals,times,t_min,t_max,t_thresh,valid = self.get_signals_and_times_from_file(shot.number,shot.t_disrupt) 
            #cut and resample
            signals,ttd = self.cut_and_resample_signals(times,signals,t_min,t_max,shot.is_disruptive)

            shot.signals = signals
            shot.ttd = ttd
            shot.valid = valid
            shot.save(processed_prepath)

        else:
            shot.restore(processed_prepath,light=True)
            print('shot {} exists.'.format(shot.number))
        shot.make_light()
        return shot 


    def get_signals_and_times_from_file(self,shot,t_disrupt):
        valid = True
        t_min = -1
        t_max = Inf
        t_thresh = -1
        signals = []
        times = []
        conf = self.conf
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



    def cut_and_resample_signals(self,times,signals,t_min,t_max,is_disruptive):
        dt = self.conf['data']['dt']
        T_max = self.conf['data']['T_max']

        #resample signals
        signals_processed = []
        assert(len(signals) == len(times) and len(signals) > 0)
        tr = 0
        for i in range(len(signals)):
            tr,sigr = cut_and_resample_signal(times[i],signals[i],t_min,t_max,dt)
            signals_processed.append(sigr)

        signals = signals_processed
        signals = np.column_stack(signals)

        if is_disruptive:
            ttd = max(tr) - tr
            ttd = np.clip(ttd,0,T_max)
        else:
            ttd = T_max*np.ones_like(tr)
        ttd = log10(ttd + 1.0*dt/10)
        return signals,ttd



class ShotList(object):
    def __init__(self,shots=None):
        self.shots = []
        if shots is not None:
            assert(all([isinstance(shot,Shot) for shot in shots]))
            self.shots = [shot for shot in shots]

    def load_from_files(self,shot_list_dir,shot_files):
        shot_numbers,disruption_times = ShotList.get_multiple_shots_and_disruption_times(shot_list_dir,shot_files)
        for number,t in zip(shot_numbers,disruption_times):
            self.append(Shot(number=number,t_disrupt=t))

            ######Generic Methods####

    @staticmethod 
    def get_shots_and_disruption_times(shots_and_disruption_times_path):
        data = loadtxt(shots_and_disruption_times_path,ndmin=1,dtype={'names':('num','disrupt_times'),
                                                                  'formats':('i4','f4')})
        shots = array(zip(*data)[0])
        disrupt_times = array(zip(*data)[1])
        return shots, disrupt_times

    @staticmethod
    def get_multiple_shots_and_disruption_times(base_path,endings):
        all_shots = []
        all_disruption_times = []
        for ending in endings:
            path = base_path + ending
            shots,disruption_times = ShotList.get_shots_and_disruption_times(path)
            all_shots.append(shots)
            all_disruption_times.append(disruption_times)
        return concatenate(all_shots),concatenate(all_disruption_times)


    def split_train_test(self,conf):
        shot_list_dir = conf['paths']['shot_list_dir']
        shot_files = conf['paths']['shot_files']
        shot_files_test = conf['paths']['shot_files_test']
        train_frac = conf['training']['train_frac']
        shuffle_training = conf['training']['shuffle_training']
        use_shots = conf['data']['use_shots']
        #split randomly
        if len(shot_files_test) == 0:
            shots_train,shots_test = train_test_split(self.shots,train_frac,shuffle_training)
            return ShotList(shots_train), ShotList(shots_test)
        #train and test list given
        else:
            use_shots_train = int(round(train_frac*use_shots))
            use_shots_test = int(round((1-train_frac)*use_shots))
            shot_numbers_train,_ = ShotList.get_multiple_shots_and_disruption_times(shot_list_dir,shot_files)
            shot_numbers_test,_ = ShotList.get_multiple_shots_and_disruption_times(shot_list_dir,shot_files_test)

            shots_train = self.filter_by_number(shot_numbers_train)
            shots_test = self.filter_by_number(shot_numbers_test)

            return shots_train.random_sublist(use_shots_train),shots_test.random_sublist(use_shots_test)


    def filter_by_number(self,numbers):
        new_shot_list = ShotList()
        numbers = set(numbers)
        for shot in self.shots:
            if shot.number in numbers:
                new_shot_list.append(shot)
        return new_shot_list

    def num_disruptive(self):
        return len([shot for shot in self.shots if shot.is_disruptive_shot()])

    def __len__(self):
        return len(self.shots) 

    def __iter__(self):
        return self.shots.__iter__()

    def next(self):
        return self.shots.next()

    def __add__(self,other_list):
        return self.shots + other_list.shots


    def random_sublist(self,num):
        num = min(num,len(self))
        shots_picked = np.random.choice(self.shots,size=num,replace=False)
        return ShotList(shots_picked)

    def sublists(self,num,shuffle=True):
        lists = []
        self.shuffle()
        for i in range(0,len(self),num):
            lists.append(self.shots[i:i+num])
        return [ShotList(l) for l in lists]



    def shuffle(self):
        shuffle(self.shots)

    def as_list(self):
        return self.shots

    def append(self,shot):
        assert(isinstance(shot,Shot))
        self.shots.append(shot)

    def append_if_valid(self,shot):
        if shot.valid:
            self.append(shot)
            return True
        else:
            print('Warning: shot {} not valid, omitting'.format(shot))
            return False

        

class Shot(object):
    def __init__(self,number=None,signals=None,ttd=None,valid=None,is_disruptive=None,t_disrupt=None):
        self.number = number #Shot number
        self.signals = signals 
        self.ttd = ttd 
        self.valid =valid 
        self.is_disruptive = is_disruptive
        self.t_disrupt = t_disrupt
        if t_disrupt is not None:
            self.is_disruptive = Shot.is_disruptive_given_disruption_time(t_disrupt)

    def __str__(self):
        string = 'number: {}\n'.format(self.number)
        string += 'signals: {}\n'.format(self.signals )
        string += 'ttd: {}\n'.format(self.ttd )
        string += 'valid: {}\n'.format(self.valid )
        string += 'is_disruptive: {}\n'.format(self.is_disruptive)
        string += 't_disrupt: {}\n'.format(self.t_disrupt)
        return string
     

    def get_number(self):
        return self.number

    def get_signals(self):
        return self.signals

    def is_valid(self):
        return self.valid

    def is_disruptive_shot(self):
        return self.is_disruptive

    def save(self,prepath):
        save_path = self.get_save_path(prepath)
        savez(save_path,number=self.number,valid=self.valid,is_disruptive=self.is_disruptive,
            signals=self.signals,ttd=self.ttd)
        print('...saved shot {}'.format(self.number))

    def get_save_path(self,prepath):
        return get_individual_shot_file(prepath,self.number,'.npz')

    def restore(self,prepath,light=False):
        assert(self.previously_saved(prepath))
        save_path = self.get_save_path(prepath)
        dat = load(save_path)

        self.number = dat['number'][()]
        self.valid = dat['valid'][()]
        self.is_disruptive = dat['is_disruptive'][()]

        if light:
            self.signals = None
            self.ttd = None 
        else:
            self.signals = dat['signals']
            self.ttd = dat['ttd']
  
    def previously_saved(self,prepath):
        save_path = self.get_save_path(prepath)
        return os.path.isfile(save_path)

    def make_light(self):
        self.signals = None
        self.ttd = None

    @staticmethod
    def is_disruptive_given_disruption_time(t):
        return t >= 0


    def load_as_X_y(self,loader,verbose=False,prediction_mode=False):
        assert(isinstance(loader,Loader))
        return loader.load_as_X_y(self,verbose=verbose,prediction_mode=prediction_mode)
        


class Loader(object):
    def __init__(self,conf,normalizer=None):
        self.conf = conf
        self.stateful = conf['model']['stateful']
        self.normalizer = normalizer
        self.verbose = True


    def load_as_X_y_list(self,shot_list,verbose=False,prediction_mode=False):
        prepath = self.conf['paths']['processed_prepath']
        signals = []
        results = []
        total_length = 0
        for shot in shot_list:
            assert(isinstance(shot,Shot))
            assert(shot.valid)
            shot.restore(prepath)

            if self.normalizer is not None:
                self.normalizer.apply(shot)
            else:
                print('Warning, no normalization. Training data may be poorly conditioned')



            if self.conf['training']['use_mock_data']:
                sig,res = self.get_mock_data()
                shot.signals = sig
                shot.ttd = res

            total_length += len(shot.ttd)
            signals.append(shot.signals)
            res = shot.ttd
            if len(res.shape) == 1:
                results.append(expand_dims(res,axis=1))
            else:
                results.append(shot.ttd)
            shot.make_light()

        sig_patches, res_patches = self.make_patches(signals,results)
        X_list,y_list = self.arange_patches(sig_patches,res_patches)

        effective_length = len(res_patches)*len(res_patches[0])
        if self.verbose:
            print('multiplication factor: {}'.format(1.0*effective_length/total_length))
            print('effective/total length : {}/{}'.format(effective_length,total_length))
            print('patch length: {} num patches: {}'.format(len(res_patches[0]),len(res_patches)))

        return X_list,y_list


    def make_deterministic_patches(self,signals,results):
        length = self.conf['model']['length']
        sig_patches = []
        res_patches = []
        min_len = self.get_min_len(signals,length)
        for sig,res in zip(signals,results):
            sig_patch, res_patch =  self.make_deterministic_patches_from_single_array(sig,res,min_len)
            sig_patches += sig_patch
            res_patches += res_patch
        return sig_patches, res_patches

    def make_deterministic_patches_from_single_array(self,sig,res,min_len):
        sig_patches = []
        res_patches = []
        assert(min_len <= len(sig))
        for start in range(0,len(sig)-min_len,min_len):
            sig_patches.append(sig[start:start+min_len])
            res_patches.append(res[start:start+min_len])
        sig_patches.append(sig[-min_len:])
        res_patches.append(res[-min_len:])
        return sig_patches,res_patches

    def make_random_patches(self,signals,results,num):
        length = self.conf['model']['length']
        sig_patches = []
        res_patches = []
        min_len = self.get_min_len(signals,length)
        for i in range(num):
            idx= np.random.randint(len(signals))
            sig_patch, res_patch =  self.make_random_patch_from_array(signals[idx],results[idx],min_len)
            sig_patches.append(sig_patch)
            res_patches.append(res_patch)
        return sig_patches,res_patches

    def make_random_patch_from_array(self,sig,res,min_len):
        start = np.random.randint(len(sig) - min_len+1)
        return sig[start:start+min_len],res[start:start+min_len]
        

    def get_min_len(self,arrs,length):
        length = self.conf['model']['length']
        min_len = min([len(a) for a in arrs] + [self.conf['training']['max_patch_length']])
        min_len = max(1,min_len // length) * length 
        return min_len

    def make_patches(self,signals,results):
        length = self.conf['model']['length']
        total_num = self.conf['training']['batch_size'] 
        sig_patches_det,res_patches_det = self.make_deterministic_patches(signals,results)
        num_already = len(sig_patches_det)
        
        total_num = int(ceil(1.0 * num_already / total_num)) * total_num
        
        
        num_additional = total_num - num_already
        assert(num_additional >= 0)
        sig_patches_rand,res_patches_rand = self.make_random_patches(signals,results,num_additional)
        if self.verbose:
            print('random to deterministic ratio: {}/{}'.format(num_additional,num_already))
        return sig_patches_det + sig_patches_rand,res_patches_det + res_patches_rand 


    def arange_patches(self,sig_patches,res_patches):
        length = self.conf['model']['length']
        batch_size = self.conf['training']['batch_size']
        assert(len(sig_patches) % batch_size == 0) #fixed number of batches
        assert(len(sig_patches[0]) % length == 0) #divisible by length
        num_batches = len(sig_patches) / batch_size
        patch_length = len(sig_patches[0])

        zipped = zip(sig_patches,res_patches)
        shuffle(zipped)
        sig_patches, res_patches = zip(*zipped) 
        X_list = []
        y_list = []
        for i in range(num_batches):
            X,y = self.arange_patches_single(sig_patches[i*batch_size:(i+1)*batch_size],
                                        res_patches[i*batch_size:(i+1)*batch_size])
            X_list.append(X)
            y_list.append(y)
        return X_list,y_list

    def arange_patches_single(self,sig_patches,res_patches):
        length = self.conf['model']['length']
        batch_size = self.conf['training']['batch_size']
        return_sequences = self.conf['model']['return_sequences']

        assert(len(sig_patches) == batch_size)
        assert(len(sig_patches[0]) % length == 0)
        num_chunks = len(sig_patches[0]) / length
        num_signals = sig_patches[0].shape[1]
        if len(res_patches[0].shape) == 1:
            num_answers = 1
        else:
            num_answers = res_patches[0].shape[1]
        
        X = zeros((num_chunks*batch_size,length,num_signals))
        if return_sequences:
            y = zeros((num_chunks*batch_size,length,num_answers))
        else:
            y = zeros((num_chunks*batch_size,num_answers))

        
        for chunk_idx in range(num_chunks):
            src_start = chunk_idx*length
            src_end = (chunk_idx+1)*length
            for batch_idx in range(batch_size):
                X[chunk_idx*batch_size + batch_idx,:,:] = sig_patches[batch_idx][src_start:src_end]
                if return_sequences:
                    y[chunk_idx*batch_size + batch_idx,:,:] = res_patches[batch_idx][src_start:src_end]
                else:
                    y[chunk_idx*batch_size + batch_idx,:] = res_patches[batch_idx][src_end-1]
        return X,y

    def load_as_X_y(self,shot,verbose=False,prediction_mode=False):
        assert(isinstance(shot,Shot))
        assert(shot.valid)
        prepath = self.conf['paths']['processed_prepath']
        return_sequences = self.conf['model']['return_sequences']
        shot.restore(prepath)

        if self.normalizer is not None:
            self.normalizer.apply(shot)
        else:
            print('Warning, no normalization. Training data may be poorly conditioned')

        signals = shot.signals
        ttd = shot.ttd

        if self.conf['training']['use_mock_data']:
            signals,ttd = self.get_mock_data()

        # if not self.stateful:
        #     X,y = self.array_to_path_and_external_pred(signals,ttd)
        # else:
        X,y = self.array_to_path_and_external_pred_cut(signals,ttd,
                return_sequences=return_sequences,prediction_mode=prediction_mode)

        shot.make_light()
        return  X,y#X,y

    def get_mock_data(self):
        signals = linspace(0,4*pi,10000)
        rand_idx = randint(6000)
        lgth = randint(1000,3000)
        signals = signals[rand_idx:rand_idx+lgth]
        #ttd[-100:] = 1
        signals = vstack([signals]*8)
        signals = signals.T
        signals[:,0] = 0.5 + 0.5*sin(signals[:,0])
        signals[:,1] = 0.5# + 0.5*cos(signals[:,1])
        signals[:,2] = 0.5 + 0.5*sin(2*signals[:,2])
        signals[:,3:] *= 0
        offset = 100
        ttd = 0.0*signals[:,0]
        ttd[offset:] = 1.0*signals[:-offset,0]
        mask = ttd > mean(ttd)
        ttd[~mask] = 0
        #mean(signals[:,:2],1)
        return signals,ttd

    def array_to_path_and_external_pred_cut(self,arr,res,return_sequences=False,prediction_mode=False):
        length = self.conf['model']['length']
        skip = self.conf['model']['skip']
        if prediction_mode:
            length = self.conf['model']['pred_length']
            if not return_sequences:
                length = 1
            skip = length #batchsize = 1!
        assert(shape(arr)[0] == shape(res)[0])
        num_chunks = len(arr) // length
        arr = arr[-num_chunks*length:]
        res = res[-num_chunks*length:]
        assert(shape(arr)[0] == shape(res)[0])
        X = []
        y = []
        i = 0

        chunk_range = range(num_chunks-1)
        i_range = range(1,length+1,skip)
        if prediction_mode:
            chunk_range = range(num_chunks)
            i_range = range(1)


        for chunk in chunk_range:
            for i in i_range:
                start = chunk*length + i
                assert(start + length <= len(arr))
                X.append(arr[start:start+length,:])
                if return_sequences:
                    y.append(res[start:start+length])
                else:
                    y.append(res[start+length-1:start+length])
        X = array(X)
        y = array(y)
        if len(shape(X)) == 1:
            X = expand_dims(X,axis=len(shape(X)))
        if return_sequences:
            y = expand_dims(y,axis=len(shape(y)))
        return X,y

    @staticmethod
    def get_batch_size(batch_size,prediction_mode):
        if prediction_mode:
            return 1
        else:
            return batch_size#Loader.get_num_skips(length,skip)

    @staticmethod
    def get_num_skips(length,skip):
        return 1 + (length-1)//skip

    # def produce_indices(signals_list):
    #     indices_list = []
    #     for xx in signals_list:
    #         indices_list.append(arange(len(xx)))
    #     return indices_list



    # def array_to_path_and_external_pred(self,arr,res,return_sequences=False):
    #     length = self.conf['model']['length']
    #     skip = self.conf['model']['skip']
    #     assert(shape(arr)[0] == shape(res)[0])
    #     X = []
    #     y = []
    #     i = 0
    #     while True:
    #         pred = i+length
    #         if pred > len(arr):
    #             break
    #         X.append(arr[i:i+length,:])
    #         if return_sequences:
    #             y.append(res[i:i+length])
    #         else:
    #             y.append(res[i+length-1])
    #         i += skip
    #     X = array(X)
    #     y = array(y)
    #     if len(shape(X)) == 1:
    #         X = expand_dims(X,axis=len(shape(X)))
    #     if return_sequences and len(shape(y)) == 1:
    #         y = expand_dims(y,axis=len(shape(y)))
    #     return X,y

   
    # def array_to_path_and_next(self,arr):
    #     length = self.conf['model']['length']
    #     skip = self.conf['model']['skip']
    #     X = []
    #     y = []
    #     i = 0
    #     while True:
    #         pred = i+length
    #         if pred >= len(arr):
    #             break
    #         X.append(arr[i:i+length])
    #         y.append(arr[i+length])
    #         i += skip
    #     X = array(X)
    #     X = expand_dims(X,axis=len(shape(X)))
    #     return X,array(y)

    
    # def array_to_path(self,arr):
    #     length = self.conf['model']['length']
    #     skip = self.conf['model']['skip']
    #     X = []
    #     i = 0
    #     while True:
    #         pred = i+length
    #         if pred > len(arr):
    #             break
    #         X.append(arr[i:i+length,:])
    #         i += skip
    #     X = array(X)
    #     if len(shape(X)) == 1:
    #         X = expand_dims(X,axis=len(shape(X)))
    #     return X

    
    #unused: handling sequences of shots        
    # def load_shots_as_X_y(conf,shots,verbose=False,stateful=True,prediction_mode=False):
    #     X,y = zip(*[load_shot_as_X_y(conf,shot,verbose,stateful,prediction_mode) for shot in shots])
    #     return vstack(X),hstack(y)

    # def load_shots_as_X_y_list(conf,shots,verbose=False,stateful=True,prediction_mode=False):
    #     return [load_shot_as_X_y(conf,shot,verbose,stateful,prediction_mode) for shot in shots]

      






######################################################
######################UTILITIES######################
######################################################
def resample_signal(t,sig,tmin,tmax,dt):
    order = argsort(t)
    t = t[order]
    sig = sig[order]
    tt = arange(tmin,tmax,dt)
    f = UnivariateSpline(t,sig,s=0,k=1,ext=0)
    sig_interp = f(tt)

    if(any(isnan(sig_interp))):
        print("signals contains nan")
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

def get_individual_shot_file(prepath,shot_num,ext='.txt'):
    return prepath + str(shot_num) + ext 

def append_to_filename(path,to_append):
    ending_idx = path.rfind('.')
    new_path = path[:ending_idx] + to_append + path[ending_idx:]
    return new_path

def train_test_split(x,frac,shuffle_data=False):
    if not isinstance(x,ndarray):
        return train_test_split_robust(x,frac,shuffle_data)
    mask = array(range(len(x))) < frac*len(x)
    if shuffle_data:
        shuffle(mask)
    return x[mask],x[~mask]

def train_test_split_robust(x,frac,shuffle_data=False):
    mask = array(range(len(x))) < frac*len(x)
    if shuffle_data:
        shuffle(mask)
    train = []
    test = []
    for (i,_x) in enumerate(x):
        if mask[i]:
            train.append(_x)
        if not mask[i]:
            test.append(_x)
    return train,test

def train_test_split_all(x,frac,shuffle_data=True):
    groups = []
    length = len(x[0])
    mask = array(range(length)) < frac*length
    if shuffle_data:
        shuffle(mask)
    for item in x:
        groups.append((item[mask],item[~mask]))
    return groups


######################################################
######################DEAD CODE#######################
######################################################

# def get_signals_and_ttds(signal_prepath,signals_dirs,processed_prepath,shots,
#     min_times,max_times,disruptive,T_max,dt,use_shots=3,recompute = False,as_array_of_shots=True):
#     all_signals = []
#     all_ttd = []
#     use_shots = min([use_shots,len(shots)])
#     for (j,shot_num) in enumerate(shots[:use_shots]):
#         shot = shots[j]
#         t_min = min_times[j]
#         t_max = max_times[j]
#         disruptive_shot = disruptive[j]
#         signals,ttd = get_signal_and_ttd(signal_prepath,signals_dirs,processed_prepath,shot,t_min,t_max,disruptive_shot,T_max,dt,recompute)
#         all_signals.append(signals)
#         all_ttd.append(ttd)
#         print(1.0*j/use_shots)

#     if as_array_of_shots:
#         return array(all_signals),array(all_ttd)
#     else:
#         signals = vstack(all_signals)
#         ttd = hstack(all_ttd)
#         return signals,ttd


# def load_all_shots_and_minmax_times(shot_list_dir,shot_files,signal_prepath,signals_dirs,current_index,use_shots,recompute_minmax):
#     all_shots = []
#     all_min_times = []
#     all_max_times = []
#     all_disruptive = []
#     for shot_filename in shot_files:
#         shot_path = join(shot_list_dir,shot_filename)
#         shot_and_minmax_times_path = append_to_filename(shot_path,'_minmax_times')
#         if os.path.isfile(shot_and_minmax_times_path) and not recompute_minmax:
#             print('minmax previously generated for {}, reading file'.format(shot_path))
#             shots,min_times,max_times,disruptive = read_shots_and_minmax_times_from_file(shot_and_minmax_times_path)
#         else:
#             print('generating minmax for {}'.format(shot_path))
#             shots,min_times,max_times,disruptive = get_shots_and_minmax_times(signal_prepath,signals_dirs,shot_path,
#                    current_index,use_shots,True,shot_and_minmax_times_path)
#         all_shots.append(shots)
#         all_min_times.append(min_times)
#         all_max_times.append(max_times)
#         all_disruptive.append(disruptive)
#     return hstack(all_shots), hstack(all_min_times), hstack(all_max_times), hstack(all_disruptive)



# # def get_shots_and_times(shots_and_times_path):
#     data = loadtxt(shots_and_times_path,npmin=1,dtype={'names':('num','timemin','timemax'),
#                                                               'formats':('i4','f4','f4')})
#     shots = array(zip(*data)[0])
#     min_times = array(zip(*data)[1])
#     max_times = array(zip(*data)[2])
#     return shots,min_times,max_times

# def get_shots_and_minmax_times(signal_prepath,signals_dirs,shots_and_disruption_times_path, 
#               current_index = 0,use_shots=-1,write_to_file=True,shots_and_minmax_times_path=None):
#     shots,disruption_times = get_shots_and_disruption_times(shots_and_disruption_times_path)
#     min_times = []
#     max_times = []
#     disruptive = []
#     current_dir = signals_dirs[current_index]
#     use_shots = min([use_shots,len(shots)])
#     shots = shots[:use_shots]
    
#     for (j,shot) in enumerate(shots):
#         t_min,t_max = get_t_minmax(signal_prepath,signals_dirs,shot) 
#         t_thresh = get_current_threshold_time(signal_prepath,current_dir,shot)
#         t_disrupt = disruption_times[j]
#         assert(t_thresh >= t_min)
#         assert(t_disrupt <= t_max)
#         if t_disrupt > 0:
#             assert(t_thresh < t_disrupt)
#         min_times.append(t_thresh)
#         if t_disrupt < 0:
#             disruptive.append(0)
#             max_times.append(t_max)
#         else:
#             disruptive.append(1)
#             max_times.append(t_disrupt)
#         print(1.0*j/use_shots)
#     min_times = array(min_times)
#     max_times = array(max_times)
#     disruptive = array(disruptive)
#     if write_to_file:
#         if shots_and_minmax_times_path == None:
#             print("Not writing out file, no path given.")
#         else:
#             write_shots_and_minmax_times_to_file(shots,min_times,max_times,disruptive,shots_and_minmax_times_path)
#     return shots,min_times,max_times,disruptive

# def write_shots_and_minmax_times_to_file(shots,min_times,max_times,disruptive,shots_and_minmax_times_path):
#     savetxt(shots_and_minmax_times_path,vstack((shots,min_times,max_times,disruptive)).transpose(), fmt='%i %f %f %i')   
    
# def read_shots_and_minmax_times_from_file(shots_and_minmax_times_path):
#     data = loadtxt(shots_and_minmax_times_path,ndmin=1,dtype={'names':('num','min_times','max_times','disruptive'),
#                                                               'formats':('i4','f4','f4','i4')})
#     shots = array(zip(*data)[0])
#     min_times = array(zip(*data)[1])
#     max_times = array(zip(*data)[2])
#     disruptive = array(zip(*data)[3])
#     return shots, min_times, max_times,disruptive

# def get_t_minmax(signal_prepath,signals_dirs,shot):
#     t_min = -1
#     t_max = Inf
#     for (i,dirname) in enumerate(signals_dirs):
#         data = loadtxt(signal_prepath+dirname + '/' + str(shot) + '.txt')
#         t = data[:,0]
#         t_min = max(t_min,t[0])
#         t_max = min(t_max,t[-1])
#     return t_min, t_max

# def get_current_threshold_time(signal_prepath,current_dir,shot):
#     current_thresh = 750000
#     data = loadtxt(signal_prepath+current_dir + '/' + str(shot) + '.txt')
#     t = data[:,0]
#     I = data[:,1]
#     assert(any(abs(I) > current_thresh))
#     index_thresh = argwhere(abs(I) > current_thresh)[0][0]
#     t_thresh = t[index_thresh]
#     return t_thresh







# def get_normalizations_for_signals(times,signals,t_min,t_max,is_disruptive,conf):
#     dt = conf['data']['dt']
#     T_max = conf['data']['T_max']

#     #resample signals
#     signals_processed = []
#     assert(len(signals) == len(times) and len(signals) > 0)
#     tr = 0
#     for i in range(len(signals)):
#         tr,sigr = cut_and_resample_signal(times[i],signals[i],t_min,t_max,dt)
#         signals_processed.append(sigr)

#     signals = signals_processed
#     signals = np.column_stack(signals)
#     standard_deviations = std(signals,0)
#     return standard_deviations






# def bool_to_int(predicate):
#     return 1 if predicate else 0

# def time_is_disruptive(t):
#     return 1 if t >= 0 else 0 

# def times_are_disruptive(ts):
#     return array([time_is_disruptive(t) for t in ts])

# def load_or_preprocess_all_shots_from_files(conf,shot_list_dir,shot_files):
   
#     shots,disruption_times = get_multiple_shots_and_disruption_times(shot_list_dir,shot_files)

#     dt = conf['data']['dt']
#     processed_prepath = conf['paths']['processed_prepath']
#     recompute = conf['data']['recompute']
#     use_shots = min([conf['data']['use_shots'],len(shots)])

#     all_signals = []
#     all_ttd = []
#     disruptive = []
#     used_shots = []
    
#     for (j,shot) in enumerate(shots[:use_shots]):
#         shot = shots[j]
#         load_file_path = get_individual_shot_file(processed_prepath,shot,'.npz')
#         if os.path.isfile(load_file_path) and not recompute:
#             print('loading shot {}'.format(shot))
#             dat = load(load_file_path)
#             signals = dat['signals']
#             ttd = dat ['ttd']
#             is_disruptive = dat ['is_disruptive']
#             valid = dat['valid']
#         else:
#             print('(re)computing shot {}'.format(shot))
#             t_disrupt = disruption_times[j]
#             is_disruptive =  t_disrupt >= 0
#           #get minmax times
#             signals,times,t_min,t_max,t_thresh,valid = get_signals_and_times_from_file(shot,t_disrupt,conf) 
#             #cut and resample
#             signals,ttd = cut_and_resample_signals(times,signals,t_min,t_max,is_disruptive,conf)

#             savez(load_file_path,signals = signals,ttd = ttd,is_disruptive=is_disruptive,valid = valid)
#             print('saved shot {}'.format(shot))

#         if valid:
#             disruptive.append(1 if is_disruptive else 0)
#             all_signals.append(signals)
#             all_ttd.append(ttd)
#             used_shots.append(shot)
#             print(1.0*j/use_shots)
#         else:
#             print('Shot {} not valid, omitting.'.format(shot))
#     return array(all_signals),array(all_ttd),array(disruptive),array(used_shots)


# def load_or_preprocess_all_shots(conf):
#     shot_files = conf['paths']['shot_files']
#     shot_list_dir = conf['paths']['shot_list_dir']
#     return load_all_shots_from_files(conf,shot_list_dir,shot_files)
  
# def get_signal_and_ttd(signal_prepath,signals_dirs,processed_prepath,shot,t_min,t_max,disruptive,T_max,dt,recompute = False):
#     load_file_path = get_individual_shot_file(processed_prepath,shot,'.npz')
#     if os.path.isfile(load_file_path) and not recompute:
#         print('loading shot {}'.format(shot))
#         dat = load(load_file_path)
#         signals = dat['signals']
#         ttd = dat ['ttd']
#     else:
#         print('(re)computing shot {}'.format(shot))
#         signals = []
#         times = []
#         for (i,dirname) in enumerate(signals_dirs):
#             data = loadtxt(get_individual_shot_file(signal_prepath+dirname + '/',shot))
#             t = data[:,0]
#             sig = data[:,1]
#             tr,sigr = cut_and_resample_signal(t,sig,t_min,t_max,dt)
#             signals.append(sigr)
#             times.append(tr)
#         signals = np.column_stack(signals)
#         signals = whiten(signals)
#         if disruptive:
#             ttd = max(tr) - tr
#             ttd = clip(ttd,0,T_max)
#         else:
#             ttd = T_max*np.ones_like(tr)
#         ttd = log10(ttd + 1.0*dt/10)
#         savez(load_file_path,signals = signals,ttd = ttd)
#         print('saved shot {}'.format(shot))

#     return signals,ttd



