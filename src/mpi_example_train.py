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

#system
from __future__ import print_function
import math,os,sys,time,datetime,os.path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functools import partial

from mpi_model import MPIModel,print_all,print_unique

#import keras sequentially because it otherwise reads from ~/.keras/keras.json with too many threads.
#from mpi_launch_tensorflow import get_mpi_task_index 
from mpi4py import MPI
comm = MPI.COMM_WORLD
task_index = comm.Get_rank()
num_workers = comm.Get_size()
NUM_GPUS = 4
MY_GPU = task_index % NUM_GPUS
base_compile_dir = '/scratch/jk7/tmp/{}'.format(task_index)
os.environ['THEANO_FLAGS'] = 'device=gpu{},floatX=float32,base_compiledir={}'.format(MY_GPU,base_compile_dir)#,mode=NanGuardMode'
import theano
#import keras
for i in range(num_workers):
  comm.Barrier()
  if i == task_index:
    print('[{}] importing Keras'.format(task_index))
    from keras import backend as K
    from keras.layers import Input,Dense, Dropout
    from keras.layers.recurrent import LSTM
    from keras.layers.wrappers import TimeDistributed
    from keras.models import Model
    from keras.optimizers import SGD



data_dir = '/tigress/jk7/tmp/data'



def get_model(batch_size = 32,num_layers = 2,hidden_units=100,num_output=1,dropout=0.1,timesteps = 100, featurelen=1,is_training=True):

    input_tensor = Input(batch_shape=(batch_size,timesteps,featurelen))
    recurrent_layer = LSTM(hidden_units,return_sequences=True,stateful = True)(input_tensor)
    output_tensor = TimeDistributed(Dense(num_output,activation='linear'))(recurrent_layer)

    model = Model(input =input_tensor,output=output_tensor)
    #model.compile(optimizer=SGD(lr=DUMMY_LR),loss='mse')

    return model


def batch_iterator(batch_size=32,timesteps = 10,multiplier=1000,epoch_length=1000,featurelen = 1):
  lag = 50
  density = 0.005
  mode = 2
  batch_shape = (batch_size,multiplier*timesteps,featurelen)
  global_step = 0
  while True:
    if mode == 1:
      xx = np.random.binomial(1,density,batch_shape)
      yy = np.zeros((batch_size,multiplier*timesteps,2))
      for i in xrange(batch_size):
        yy[i,:,0] = turn_array_into_switch(xx[i,:,0])
        yy[i,:,1] = 1.0 - turn_array_into_switch(xx[i,:,0])
      yy = np.roll(yy,lag,axis=1)
      for chunk_idx in xrange(multiplier):
        epoch_end = global_step == epoch_length - 1
        num_so_far = global_step
        num_total = epoch_length
        reset_states_now = chunk_idx == 0
        start = chunk_idx*timesteps
        stop = (1+chunk_idx)*timesteps
        x_batch = xx[:,start:stop,:]
        y_batch = yy[:,start:stop,:]
        global_step += 1
        yield x_batch,y_batch,reset_states_now,num_so_far,num_total


    if mode == 2:
      xx = 1.0/np.sqrt(multiplier*timesteps)*np.random.randn(batch_size,multiplier*timesteps+lag,featurelen) 
      xx = np.cumsum(xx,axis=1)
      #xx = xx/np.max(np.abs(xx))
      for i in xrange(batch_size):
        xx[i,:,:] = np.roll(xx[i,:,:],np.random.randint(0,multiplier*timesteps+lag),axis=0)
      for chunk_idx in xrange(multiplier):
        epoch_end = global_step == epoch_length - 1
        num_so_far = global_step
        num_total = epoch_length
        reset_states_now = chunk_idx == 0
        start = chunk_idx*timesteps
        stop = (1+chunk_idx)*timesteps
        x_batch = xx[:,start+lag:stop+lag,:]
        y_batch = xx[:,start:stop,:]
        global_step += 1
        yield x_batch,y_batch,reset_states_now,num_so_far,num_total


def turn_array_into_switch(arr):
  out_arr = np.zeros_like(arr)
  current = 0.0
  for i in range(len(arr)):
    if arr[i] > 0.5:
      current = 1.0 - current
    out_arr[i] = current
  return out_arr





def test(model,batch_size=1,epoch=None):
  ys_pred_list = []
  xs_list = []
  ys_true_list = []
  num_concat = 200
  multiplier = num_concat 
  for i,(batch_xs,batch_ys,_,_) in enumerate(batch_iterator(batch_size=batch_size,multiplier=multiplier)):
    if i >= num_concat:
      break

    ys_pred_list.append(model.predict(batch_xs,batch_size=batch_size))
    xs_list.append(batch_xs)
    ys_true_list.append(batch_ys)
  ys_pred = np.squeeze(np.concatenate(ys_pred_list,axis=1))
  xs = np.squeeze(np.concatenate(xs_list,axis=1))
  ys_true = np.squeeze(np.concatenate(ys_true_list,axis=1))
  print('Testing loss: {}'.format(np.mean((ys_pred - ys_true)**2)))

  plt.close('all')
  plt.figure()
  plt.plot(xs,'b')
  plt.plot(ys_pred,'r')
  plt.plot(ys_true,'g')
  #plt.ylim([-0.1,1.1])
  plt.show()
  if epoch is None:
    epoch = time.time()
  plt.savefig('out_{}.png'.format(epoch),bbox_inches='tight')






def main():
  save_path = 'tmp_mpi/model_weights_epoch{}.h5'#{}.h5'
  warmup_steps = 500
  train_steps = 500
  epochs = 20
  lr = 0.01
  lr_decay = 0.8
  batch_size = 512

  loss = 'mse'
  

  hidden_units = 100
  multiplier = 50
  timesteps = 10
  num_replicas = 5

  print_all('Building model\n')
  model = get_model(batch_size=batch_size,timesteps=timesteps)
  batch_it = partial(batch_iterator,batch_size=batch_size,timesteps = timesteps,multiplier=multiplier,epoch_length=train_steps)
  mpi_model = MPIModel(model,comm,batch_it,batch_size,lr=lr,warmup_steps=warmup_steps,num_replicas=num_replicas)
  mpi_model.compile(loss=loss)

  for e in range(epochs):
    mpi_model.set_lr(lr*lr_decay**e)
    print_unique('\nEpoch {}\n'.format(e))


    if task_index == 0:
      print('Evaluating model...')
      save_path_curr = save_path.format(e)
      mpi_model.save_weights(save_path_curr,overwrite=True)
      test_model = get_model(batch_size = 1,timesteps=timesteps)
      test_model.load_weights(save_path_curr)
      test(test_model,epoch=e)
      print('done.')



    mpi_model.train_epoch()




if __name__ == "__main__":
  main() 


