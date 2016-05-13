
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')

import datetime
from data_processing import *
from model_builder import build_model
import numpy as np
import os.path
from conf import conf


shot_list_dir = conf['paths']['shot_list_dir']
shot_files = conf['paths']['shot_files']
print("Clean Shot Lists",end='')
clean_shots_lists(shot_list_dir)
print("...done")

#signals_by_shot,ttd_by_shot,disruptive = load_all_shots(conf)

print("preprocessing shots",end='')
shots,disruption_times = preprocess_all_shots(conf)
print("...done")

use_shots = min(conf['data']['use_shots'],len(shots))
shots = shots[:use_shots]
disruptive = times_are_disruptive(disruption_times)[:use_shots]


T_warning = conf['data']['T_warning']
train_frac = conf['training']['train_frac']
shuffle_training = conf['training']['shuffle_training']
num_epochs = conf['training']['num_epochs']
batch_size_large = conf['training']['batch_size_large']
batch_size_small = conf['training']['batch_size_small']
num_shots_at_once = conf['training']['num_shots_at_once']
length = conf['model']['length']
skip = conf['model']['skip']

split_groups = train_test_split_all((shots,disruptive),train_frac,shuffle_training)
shots_train,shots_test = split_groups[0]
disruptive_train,disruptive_test = split_groups[1]

num_shots = len(shots)
num_shots_train = len(shots_train)
num_shots_test = len(shots_test)



print('Build model...',end='')
model = build_model(conf)
print('...done')


print('training model')
for e in range(num_epochs):
    #train on small batches in first and last epoch
    if e == 0 or e == num_epochs - 1:
        batch_size = batch_size_small
    #otherwise train on large batches
    else:
        batch_size = batch_size_large 
    shots_arrays = array_split(np.random.permutation(shots_train),int(round(1.0*num_shots_train/num_shots_at_once)))
    print('Epoch {}/{}'.format(e+1,num_epochs))
    for i,shots_array in enumerate(shots_arrays):
        X_train,y_train = load_shots_as_X_y(conf,shots_array)
        print('Shots {}/{}'.format(len(shots_array)*(i+1),num_shots_train))
        model.fit(X_train,y_train,batch_size=batch_size,nb_epoch=1,verbose=1,validation_split=0.0)
print('...done')

print('evaluating model')
for (i,shot) in enumerate(shots_test):
    print('Shot {}/{}'.format(shot_idx,num_shots_test))
    X,y = load_shot_as_X_y(conf,shot)
    res = model.evaluate(X,y)
    print(res)
print('...done')


print('saving results')
y_prime = []
y_prime_test = []
y_prime_train = []

y_gold = []
y_gold_test = []
y_gold_train = []

for (i,shot) in enumerate(shots_train):
    print('Shot {}/{}'.format(i,2*num_shots))
    X,y = load_shot_as_X_y(conf,shot)
    y_prime_train.append(model.predict(X,batch_size=batch_size_large))
    y_gold_train.append(y)

for (i,shot) in enumerate(shots_test):
    print('Shot {}/{}'.format(i,2*num_shots))
    X,y = load_shot_as_X_y(conf,shot)
    y_prime_test.append(model.predict(X,batch_size=batch_size_large))
    y_gold_test.append(y)

for (i,shot) in enumerate(shots):
    print('Shot {}/{}'.format(i,2*num_shots))
    X,y = load_shot_as_X_y(conf,shot)
    y_prime.append(model.predict(X,batch_size=batch_size_large))
    y_gold.append(y)


save_str = 'results_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
savez(conf['paths']['results_prepath']+save_str,
    y_gold=y_gold,y_gold_train=y_gold_train,y_gold_test=y_gold_test,
    y_prime=y_prime,y_prime_train=y_prime_train,y_prime_test=y_prime_test,
    disruptive=disruptive,disruptive_train=disruptive_train,disruptive_test=disruptive_test)


if plotting:
    print('plotting results')
    plot(ttd)
    plot(ttd_prime)
    plot(indices_test,ttd_prime_test,'g')
    plot(indices_train,ttd_prime_train,'r')
    savefig('plot.png')
    #plot(y_train,'.')
    #show()

print('finished.')





