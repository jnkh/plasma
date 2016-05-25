from __future__ import print_function
 
import sys
import numpy as np
import random
import time
 
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.generic_utils import Progbar 
from keras.callbacks import Callback
 
from data_processing import *
from pylab import *


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

def build_model(predict,batch_size,length,featurelen):
    if predict:
        batch_size = length = 1
    model = Sequential()
    model.add(LSTM(10 ,return_sequences=True, batch_input_shape=(batch_size, length , featurelen), stateful=True))
    model.add(Dropout(0.2))
    model.add(LSTM(10 , return_sequences=True,stateful=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense( featurelen )))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.reset_states()
    return model


def num_to_one_hot_vec(i,vocab_size):
    arr = zeros(vocab_size)
    arr[i] = 1
    return arr

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



allChars= ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o',
           'p','q','r','s','t','u','v','w','x','y','z','y','x','w','v',
           'u','t','s','r','q','p','o','m','n','l','k','j','i','h','g',
           'f','e','d','c','b']





allChars *= 10
num_shots = 100
raw_shots = [allChars for _ in range(num_shots)]

length = 20
batch_size = length
skip = 1
return_sequences = True

print('shot length:', len(allChars))
print('num shots:', num_shots)
chars = sorted(list(set(allChars)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
featurelen = len(chars)



all_shots = []
for raw_shot in raw_shots:
    text_num = array([char_indices[c] for c in raw_shot])
    text_array = array([num_to_one_hot_vec(i,featurelen) for i in text_num]) 
    all_shots.append(text_array)


print('Building model...')
 
model = build_model(False,batch_size,length,featurelen)
testing_model = build_model(True,batch_size,length,featurelen)

 
print('starting training')
num_epochs = 3
for e in range(num_epochs):
    print('epoch - ',e+1)
    startTime = time.time()
    pbar =  Progbar(len(all_shots))
    for (idx,shot) in enumerate(all_shots):
    	# print('Shot: {}/{}'.format(idx,len(all_shots)))
    	history = LossHistory()
        X,y = array_to_path_and_external_pred_cut(shot[:-1,:],shot[1:,:],length,skip,return_sequences)
        model.fit(X,y,batch_size=batch_size,nb_epoch=1,verbose=0,callbacks=[history])
#     for i in range(0,totalTimeSteps-1):
#         model.train_on_batch(X[:, numOfPrevSteps*i:(i+1)*numOfPrevSteps, :], np.reshape(X[:, (i+1)*numOfPrevSteps, :], (batchSize, featurelen)) ) # Train on guessing a single element based on the previous element
        model.reset_states()
        pbar.add(1, values=[("train loss", history.losses[-1])])
    
    model.save_weights('./tmp/keras_char_rnn.%d.h5' % e,overwrite=True)
    numberToGenerate = 100
    diversities = [.2,.5,1.0,1.2]
    startChar = np.random.choice(allChars)
    testing_model.load_weights('./tmp/keras_char_rnn.%d.h5' % e)
    for diversity in diversities:
        next_char = startChar
        print('Generating with diversity - ',diversity,' and seed - ',next_char)
        for _ in range(numberToGenerate):
            C = np.zeros([1,1,featurelen])
            C[0][0][char_indices[next_char]] = 1
            pred = testing_model.predict(C)
            pred = pred[0,0,:] # We have to pass in an entire batch at a time, but we only care about the first since we are only generation one letter
            next_index = sample(pred, diversity)
            next_char = indices_char[next_index]
            sys.stdout.write(next_char)
            sys.stdout.flush()
        testing_model.reset_states()
        print()
    totalTime = time.time() - startTime
    print('Completed epoch in ',totalTime,' seconds')
    print()
print('training complete')