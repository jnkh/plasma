
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')

import datetime
import time
from data_processing import *
from model_builder import build_model
import numpy as np
import os.path
from conf import conf
from pprint import pformat


from keras.utils.generic_utils import Progbar 
from keras.callbacks import Callback


shot_list_dir = conf['paths']['shot_list_dir']
shot_files = conf['paths']['shot_files']
shot_files_test = conf['paths']['shot_files_test']
train_frac = conf['training']['train_frac']

print("Clean Shot Lists",end='')
clean_shots_lists(shot_list_dir)
print("...done")

#signals_by_shot,ttd_by_shot,disruptive = load_all_shots(conf)

#only one list of shots -- split randomly
if len(shot_files_test) == 0:
    print("preprocessing all shots",end='')
    shots,disruptive = preprocess_all_shots(conf)
    print("...done")

    split_groups = train_test_split_all((shots,disruptive),train_frac,conf['training']['shuffle_training'])
    shots_train,shots_test = split_groups[0]
    disruptive_train,disruptive_test = split_groups[1]

#train and test list given
else:
    use_shots_train = int(round(train_frac*conf['data']['use_shots']))
    use_shots_test = int(round((1-train_frac)*conf['data']['use_shots']))
    print("preprocessing training shots",end='')
    shots_train,disruptive_train = preprocess_all_shots_from_files(conf,shot_list_dir,shot_files,use_shots_train)
    print("preprocessing testing shots",end='')
    shots_test,disruptive_test = preprocess_all_shots_from_files(conf,shot_list_dir,shot_files_test,use_shots_test)
    print("...done")

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

    
num_shots_train = len(shots_train)
num_shots_test = len(shots_test)
num_shots = num_shots_train + num_shots_test

print('Training on {} shots, testing on {} shots'.format(num_shots_train,num_shots_test))



print('Build model...',end='')
train_model = build_model(conf,False)
test_model = build_model(conf,True)
print('...done')

unique_configuration_id = hash(pformat(conf))
print('training model')
for e in range(num_epochs):
    # #train on small batches in first and last epoch
    # if e == 0 or e == num_epochs - 1:
    #     batch_size = batch_size_small
    # #otherwise train on large batches
    # else:
    #     batch_size = batch_size_large 
    batch_size = length
    shots_array = np.random.permutation(shots_train)
    # shots_arrays = array_split(np.random.permutation(shots_train),max(1,int(round(1.0*num_shots_train/num_shots_at_once))))
    print('Epoch {}/{}'.format(e+1,num_epochs))
    # for i,shots_array in enumerate(shots_arrays):
    for (i,shot) in enumerate(shots_array):
        print('Shots {}/{}'.format(i,len(shots_array)))
        start = time.time()
        X_train,y_train = load_shot_as_X_y(conf,shot,stateful=True)
        print('loaded shot in {} seconds'.format(time.time() - start))
        pbar =  Progbar(len(shots_array))

        # print('Shots {}/{}'.format(len(shots_array)*(i+1),num_shots_train))
        # for (X_train,y_train) in X_y_train_list:
        train_model.reset_states()
        history = LossHistory()
        train_model.fit(X_train,y_train,batch_size=batch_size,nb_epoch=1,verbose=1,validation_split=0.0,callbacks=[history])
        pbar.add(1, values=[("train loss", mean(history.losses))])

    train_model.save_weights('./tmp/train_model.{}._epoch_.{}.h5'.format(unique_configuration_id,e),overwrite=True)

print('...done')

#load last model for testing
test_model.load_weights('./tmp/train_model.{}._epoch_.{}.h5'.format(unique_configuration_id,(num_epochs-1)))

if conf['training']['evaluate']:
    print('evaluating model')
    for (i,shot) in enumerate(shots_test):
        print('Shot {}/{}'.format(i,num_shots_test))
        X,y = load_shot_as_X_y(conf,shot,stateful=True)
        res = train_model.evaluate(X,y,batch_size=batch_size_large)
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
    test_model.reset_states()
    print('Shot {}/{}'.format(i,num_shots))
    X,y = load_shot_as_X_y(conf,shot,stateful=True,prediction_mode=True)
    assert(X.shape[0] == y.shape[0])
    shot_length = X.shape[0]
    y_prime_train_curr = []
    for j in range(shot_length):
        X_row = X[j:j+1,:,:]
        y_prime_train_curr.append(test_model.predict(X_row))
    y_prime_train_curr = squeeze(vstack(y_prime_train_curr),axis=1)
    y_prime_train.append(y_prime_train_curr)
    y_gold_train.append(squeeze(y,axis=1))


for (i,shot) in enumerate(shots_test):
    test_model.reset_states()
    print('Shot {}/{}'.format(i + len(shots_train),num_shots))
    X,y = load_shot_as_X_y(conf,shot,stateful=True,prediction_mode=True)
    assert(X.shape[0] == y.shape[0])
    shot_length = X.shape[0]
    y_prime_test_curr = []
    for j in range(shot_length):
        X_row = X[j:j+1,:,:]
        y_prime_test_curr.append(test_model.predict(X_row))
    y_prime_test_curr = squeeze(vstack(y_prime_test_curr),axis=1)
    y_prime_test.append(y_prime_test_curr)
    y_gold_test.append(squeeze(y,axis=1))



 
disruptive = concatenate((disruptive_train,disruptive_test))
y_gold = concatenate((y_gold_train,y_gold_test))
y_prime = concatenate((y_prime_train,y_prime_test))

save_str = 'results_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
savez(conf['paths']['results_prepath']+save_str,
    y_gold=y_gold,y_gold_train=y_gold_train,y_gold_test=y_gold_test,
    y_prime=y_prime,y_prime_train=y_prime_train,y_prime_test=y_prime_test,
    disruptive=disruptive,disruptive_train=disruptive_train,disruptive_test=disruptive_test,
    conf = conf)


# if plotting:
#     print('plotting results')
#     plot(ttd)
#     plot(ttd_prime)
#     plot(indices_test,ttd_prime_test,'g')
#     plot(indices_train,ttd_prime_train,'r')
#     savefig('plot.png')
#     #plot(y_train,'.')
#     #show()

print('finished.')





