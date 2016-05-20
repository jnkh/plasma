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
import datetime,time,os.path
from pprint import pformat

#matplotlib
import matplotlib
matplotlib.use('Agg')
import numpy as np


#keras

#my classes
from conf import conf
from data_processing import *


num_epochs = conf['training']['num_epochs']
shot_list_dir = conf['paths']['shot_list_dir']
shot_files = conf['paths']['shot_files']
shot_files_test = conf['paths']['shot_files_test']
train_frac = conf['training']['train_frac']
stateful = conf['model']['stateful']
if stateful: 
    batch_size = conf['model']['length']
else:
    batch_size = conf['training']['batch_size_large']



print("preprocessing all shots",end='')
pp = Preprocessor(conf)
pp.clean_shot_lists()
shot_list = pp.preprocess_all()
shot_list_train,shot_list_test = shot_list.split_train_test(conf)
print("...done")

print("normalization",end='')
nn = Normalizer(conf)
nn.train()
loader = Loader(conf,nn)
print("...done")



from keras.utils.generic_utils import Progbar 
from model_builder import build_model, LossHistory

print('Build model...',end='')
train_model = build_model(conf,False)
test_model = build_model(conf,True)
print('...done')

num_shots_train = len(shot_list_train)
num_shots_test = len(shot_list_test)
num_shots = num_shots_train + num_shots_test

print('Training on {} shots, testing on {} shots'.format(num_shots_train,num_shots_test))
unique_configuration_id = hash(pformat(conf))
for e in range(num_epochs):

    #history
    print('Epoch {}/{}'.format(e+1,num_epochs))
    pbar =  Progbar(len(shot_list_train))

    #shuffle during every iteration
    shot_list_train.shuffle() 
    for (i,shot) in enumerate(shot_list_train):
        history = LossHistory()
        #load data and fit on data
        X_train,y_train = loader.load_as_X_y(shot,prediction_mode=False)
        train_model.reset_states()
        train_model.fit(X_train,y_train,batch_size=batch_size,nb_epoch=1,verbose=1,validation_split=0.0,callbacks=[history])

        print('Shots {}/{}'.format(i,len(shot_list_train)))
        pbar.add(1, values=[("train loss", mean(history.losses))])

    train_model.save_weights('./tmp/train_model.{}._epoch_.{}.h5'.format(unique_configuration_id,e),overwrite=True)

print('...done')

#load last model for testing
test_model.load_weights('./tmp/train_model.{}._epoch_.{}.h5'.format(unique_configuration_id,(num_epochs-1)))

# if conf['training']['evaluate']:
#     print('evaluating model')
#     for (i,shot) in enumerate(shots_test):
#         print('Shot {}/{}'.format(i,num_shots_test))
#         X,y = load_shot_as_X_y(conf,shot,stateful=True)
#         res = train_model.evaluate(X,y,batch_size=batch_size_large)
#         print(res)
#     print('...done')


print('saving results')
y_prime = []
y_prime_test = []
y_prime_train = []

y_gold = []
y_gold_test = []
y_gold_train = []

for (i,shot) in enumerate(shot_list_train):
    test_model.reset_states()
    print('Shot {}/{}'.format(i,num_shots))
    X,y = loader.load_as_X_y(conf,shot,stateful=True,prediction_mode=True)
    assert(X.shape[0] == y.shape[0])
    shot_length = X.shape[0]
    y_prime_train_curr = []
    for j in range(shot_length):
        X_row = X[j:j+1,:,:]
        y_prime_train_curr.append(test_model.predict(X_row))
    y_prime_train_curr = squeeze(vstack(y_prime_train_curr),axis=1)
    y_prime_train.append(y_prime_train_curr)
    y_gold_train.append(squeeze(y,axis=1))


for (i,shot) in enumerate(shot_list_test):
    test_model.reset_states()
    print('Shot {}/{}'.format(i + len(shots_train),num_shots))
    X,y = loader.load_as_X_y(conf,shot,stateful=True,prediction_mode=True)
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





