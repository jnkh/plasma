'''
#########################################################
This file trains a deep learning model to predict
disruptions on time series data from plasma discharges.

Dependencies:
conf.py: configuration of model,training,paths, and data
model_builder.py: logic to construct the ML architecture
data_processing.py: classes to handle data processing

Author: Julian Kates-Harbeck, jkatesharbeck@g.harvard.edu

This work was supported by the DOE CSGF program.
#########################################################
'''

from __future__ import print_function
import datetime,time,os.path,sys
import dill
from functools import partial

#matplotlib
import matplotlib
matplotlib.use('Agg')
import numpy as np

#my classes
from conf import conf
from pprint import pprint
pprint(conf)
from data_processing import Shot, ShotList, Normalizer, Preprocessor, Loader

if conf['data']['normalizer'] == 'minmax':
    from data_processing import MinMaxNormalizer as Normalizer #performs !much better than minmaxnormalizer
elif conf['data']['normalizer'] == 'meanvar':
    from data_processing import MeanVarNormalizer as Normalizer #performs !much better than minmaxnormalizer
elif conf['data']['normalizer'] == 'var':
    from data_processing import VarNormalizer as Normalizer #performs !much better than minmaxnormalizer
else:
    print('unkown normalizer. exiting')
    exit(1)

shot_list_dir = conf['paths']['shot_list_dir']
shot_files = conf['paths']['shot_files']
shot_files_test = conf['paths']['shot_files_test']
train_frac = conf['training']['train_frac']
stateful = conf['model']['stateful']
# if stateful: 
#     batch_size = conf['model']['length']
# else:
#     batch_size = conf['training']['batch_size_large']

np.random.seed(1)
#####################################################
####################PREPROCESSING####################
#####################################################

print("preprocessing all shots",end='')
pp = Preprocessor(conf)
pp.clean_shot_lists()
shot_list = pp.preprocess_all()
sorted(shot_list)
shot_list_train,shot_list_test = shot_list.split_train_test(conf)
num_shots = len(shot_list_train) + len(shot_list_test)
print("...done")


#####################################################
####################Normalization####################
#####################################################


print("normalization",end='')
nn = Normalizer(conf)
nn.train()
loader = Loader(conf,nn)
print("...done")


print('Training on {} shots, testing on {} shots'.format(len(shot_list_train),len(shot_list_test)))




#####################################################
######################TRAINING#######################
#####################################################
import multiprocessing as old_mp
from model_runner import train
p = old_mp.Process(target = train,args=(conf,shot_list_train,loader))
p.start()
p.join()


#####################################################
####################PREDICTING#######################
#####################################################


from model_runner import make_predictions
from model_builder import ModelBuilder
builder = ModelBuilder(conf)


#load last model for testing
print('saving results')
y_prime = []
y_prime_test = []
y_prime_train = []

y_gold = []
y_gold_test = []
y_gold_train = []

disruptive= []
disruptive_train= []
disruptive_test= []



y_prime_train,y_gold_train,disruptive_train = make_predictions(conf,shot_list_train,builder,loader)
y_prime_test,y_gold_test,disruptive_test = make_predictions(conf,shot_list_test,builder,loader)



disruptive_train = np.array(disruptive_train)
disruptive_test = np.array(disruptive_test)

y_gold = y_gold_train + y_gold_test
y_prime = y_prime_train + y_prime_test
disruptive = np.concatenate((disruptive_train,disruptive_test))

shot_list.make_light()
shot_list_test.make_light()
shot_list_train.make_light()

save_str = 'results_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
np.savez(conf['paths']['results_prepath']+save_str,
    y_gold=y_gold,y_gold_train=y_gold_train,y_gold_test=y_gold_test,
    y_prime=y_prime,y_prime_train=y_prime_train,y_prime_test=y_prime_test,
    disruptive=disruptive,disruptive_train=disruptive_train,disruptive_test=disruptive_test,
    shot_list=shot_list,shot_list_train=shot_list_train,shot_list_test=shot_list_test,
    conf = conf)



print('finished.')





