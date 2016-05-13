
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
shot_files_test = conf['paths']['shot_files_test']

print("Clean Shot Lists",end='')
clean_shots_lists(shot_list_dir)
print("...done")

#signals_by_shot,ttd_by_shot,disruptive = load_all_shots(conf)

#only one list of shots -- split randomly
if len(conf['paths']['test_shot_files']) == 0:
    print("preprocessing all shots",end='')
    shots,disruption_times = preprocess_all_shots(conf)
    print("...done")

    use_shots = min(conf['data']['use_shots'],len(shots))
    shots = np.random.choice(shots,use_shots,replace=False)
    disruptive = times_are_disruptive(disruption_times)[:use_shots]

    split_groups = train_test_split_all((shots,disruptive),train_frac,shuffle_training)
    shots_train,shots_test = split_groups[0]
    disruptive_train,disruptive_test = split_groups[1]

#train and test list given
else:
    print("preprocessing training shots",end='')
    shots_train,disruptions_times_train = preprocess_all_shots_from_files(conf,shot_list_dir,shot_files):
    print("preprocessing testing shots",end='')
    shots_test,disruptions_times_test = preprocess_all_shots_from_files(conf,shot_list_dir,shot_files_test):
    print("...done")

    use_shots_train = min(int(round(train_frac*conf['data']['use_shots'])),len(shots_train))
    use_shots_test = min(int(round((1-train_frac)*conf['data']['use_shots'])),len(shots_test))

    shots_train = np.random.choice(shots_train,use_shots_train,replace=False)
    shots_test = np.random.choice(shots_test,use_shots_test,replace=False)

    disruptive_train = times_are_disruptive(disruption_times_train)[:use_shots_train]
    disruptive_test = times_are_disruptive(disruption_times_test)[:use_shots_test]

    shots = np.concatenate((shots_train,shots_test))
    disruptive = np.concatenate((disruptive_train,disruptive_test))


T_warning = conf['data']['T_warning']
train_frac = conf['training']['train_frac']
shuffle_training = conf['training']['shuffle_training']
num_epochs = conf['training']['num_epochs']
batch_size_large = conf['training']['batch_size_large']
batch_size_small = conf['training']['batch_size_small']
num_shots_at_once = conf['training']['num_shots_at_once']
length = conf['model']['length']
skip = conf['model']['skip']

if len(conf['paths']['test_shot_files']) == 0:
    
num_shots_train = len(shots_train)
num_shots_test = len(shots_test)
num_shots = num_shots_train + num_shots_test



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
    print('Shot {}/{}'.format(i,num_shots_test))
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





