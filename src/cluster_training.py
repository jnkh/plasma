from __future__ import print_function
 
import sys
import numpy as np
import random
import time

#import tensorflow as tf 
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.generic_utils import Progbar 
from keras.callbacks import Callback
from keras.optimizers import Adam
 
from data_processing import *
from pylab import *


def build_model(predict,batch_size,length,featurelen):
    if predict:
        batch_size = length = 1
    model = Sequential()
    model.add(LSTM(10 ,return_sequences=True, batch_input_shape=(batch_size, length , featurelen), stateful=True))
    # model.add(Dropout(0.1))
    # model.add(LSTM(10 , return_sequences=True,stateful=True))
    # model.add(Dropout(0.1))
    model.add(TimeDistributed(Dense( featurelen )))


    opt = Adam(lr = 0.00001)
    model.compile(loss='mse', optimizer='adam')
    model.reset_states()
    return model


def array_to_path_and_external_pred_cut(arr,res,length,skip,return_sequences=False):
    assert(shape(arr)[0] == shape(res)[0])
    num_chunks = len(arr) // length
    arr = arr[-num_chunks*length:,:]
    res = res[-num_chunks*length:,:]
    X = []
    y = []
    i = 0
    for chunk in range(num_chunks-1):
        for i in range(1,length+1,skip):
            start = chunk*length + i
            assert(start + length <= len(arr))
            X.append(arr[start:start+length,:])
            if return_sequences:
                y.append(res[start:start+length])
            else:
                y.append(res[start+length-1])
    X = array(X)
    y = array(y)
    if len(shape(X)) == 1:
        X = expand_dims(X,axis=len(shape(X)))
    if return_sequences and len(shape(y)) == 1:
        y = expand_dims(y,axis=len(shape(y)))
    return X,y


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def get_mock_data():
    x_raw = cumsum(randn(1000))
    x_raw = vstack([x_raw]*1)
    x_raw = x_raw.T
    x_raw = 1.0*(x_raw - x_raw.min()) / (x_raw.max() - x_raw.min())
    x = x_raw[10:]
    y = x_raw[:-10]
    return x,y

def rms(y1,y2):
    return mean((y1-y2)**2)

plotting = False
num_shots = 50
length = 20
batch_size = length
skip = 1
return_sequences = True

featurelen = get_mock_data()[1].shape[1]



print('Building model...')
 
model = build_model(False,batch_size,length,featurelen)
testing_model = build_model(True,batch_size,length,featurelen)

 
print('starting training')
num_epochs = 10
for e in range(num_epochs):
    shots = [get_mock_data() for _ in range(num_shots)]
    print('epoch - ',e+1)
    startTime = time.time()
    pbar =  Progbar(len(shots))
    for (idx,shot) in enumerate(shots):
    	# print('Shot: {}/{}'.format(idx,len(all_shots)))
    	history = LossHistory()
        xx,yy = shot
        X,y = array_to_path_and_external_pred_cut(xx,yy,length,skip,return_sequences)
        model.fit(X,y,batch_size=batch_size,nb_epoch=1,shuffle=False,verbose=0,callbacks=[history])
#     for i in range(0,totalTimeSteps-1):
#         model.train_on_batch(X[:, numOfPrevSteps*i:(i+1)*numOfPrevSteps, :], np.reshape(X[:, (i+1)*numOfPrevSteps, :], (batchSize, featurelen)) ) # Train on guessing a single element based on the previous element
        model.reset_states()
        pbar.add(1, values=[("train loss", history.losses[-1])])
    

    model.save_weights('./tmp1/cluster_training.%d.h5' % e,overwrite=True)
    testing_model.load_weights('./tmp1/cluster_training.%d.h5' % e)
    y_pred = []
    xx,yy = get_mock_data()
    for i in range(len(xx)):
        C = np.zeros([1,1,featurelen])
        C[0,0,:] = xx[i,:]
        pred = testing_model.predict(C)
        pred = pred[0,0,:] # We have to pass in an entire batch at a time, but we only care about the first since we are only generation one letter
        y_pred.append(pred)
    y_pred = array(y_pred)
    totalTime = time.time() - startTime
    print('Completed epoch in ',totalTime,' seconds')
    print('===============================')
    print ('Test Loss: {}'.format(rms(y_pred,yy)))
    if plotting:
        figure(e)
        plot(xx,'-b')
        plot(yy,'-g')
        plot(y_pred,'-r')
        show()
    testing_model.reset_states()

    print()
print('training complete')