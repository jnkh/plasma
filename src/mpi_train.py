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

#import keras sequentially because it otherwise reads from ~/.keras/keras.json with too many threads.
#from mpi_launch_tensorflow import get_mpi_task_index 
from mpi4py import MPI
comm = MPI.COMM_WORLD
task_index = comm.Get_rank()
num_workers = comm.Get_size()
NUM_GPUS = 4
MY_GPU = task_index % NUM_GPUS
print('importing theano')
os.environ['THEANO_FLAGS'] = 'device=gpu{},floatX=float32'.format(MY_GPU)#,mode=NanGuardMode'
import theano

#import keras
print('Importing Keras')
for i in range(num_workers):
  mpi_comm.Barrier()
  if i == task_index:
    from keras import backend as K
    from keras.layers import Input,Dense, Dropout
    from keras.layers.recurrent import LSTM
    from keras.layers.wrappers import TimeDistributed
    from keras.models import Model


IMAGE_PIXELS = 28
hidden_units = 20
batch_size = 512
sync_mode = True
data_dir = '/tigress/jk7/tmp/data'


def get_model(batch_size = 32,timesteps = 100, featurelen=1,is_training=True):

    num_layers = 2
    num_hidden = 10
    num_output = 1
    dropout = 0.1

    input_tensor = Input()
    recurrent_layer = LSTM(hidden_units,return_sequences=True,stateful = True,batch_input_shape=(batch_size,timesteps,featurelen))(input_tensor)
    output_tensor = TimeDistributed(Dense(num_output,activation='tanh'))(recurrent_layer)

    model = Model(input =input_tensor,output=output_tensor)
    model.compile(optimizer=SGD(lr=lr),loss='mse')

    return model


def next_batch(batch_size=32,timesteps = 100,featurelen = 1):
  lag = 0
  x = np.random.randn(batch_size,timesteps+lag,featurelen) 
  x = np.cumsum(x,axis=1)
  x = x/np.max(np.abs(x))
  if lag == 0:
    return x,x
  else:
    return x[:,lag:,:],x[:,:-lag,:]

def sync_deltas(deltas):
  global_deltas = []
  for delta in deltas:
    global_deltas.append(mpi_reduce_array(delta))
  return global_deltas 

def mpi_reduce_array(arr):
  arr_global = np.empty_like(arr)
  comm.Allreduce([arr,MPI.DOUBLE],[arr_global,MPI.DOUBLE],op=MPI.SUM,root=0)
  arr_global /= num_workers
  return arr_global

def get_deltas(model,X_batch,Y_batch):
  weights_before_update = model.get_weights()
  loss = model.train_on_batch(X_batch,Y_batch)
  weights_after_update = model.get_weights()
  deltas = [w1 - w0 for w1,w0 in zip(weights_after_update,weights_before_update)]
  return deltas,loss


def apply_deltas(model,deltas):
  model.set_weights()

def get_new_weights(model,deltas):
  return [w-d for w,d in zip(model.get_weights(),deltas)]


def set_new_weights(model,deltas):
  global_deltas = sync_deltas(deltas)
  if comm.rank == 0:
    new_weights = get_new_weights(model,global_deltas)
  else:
    new_weights = None
  new_weights = comm.bcast(new_weights,root=0)
  model.set_weights(new_weights)


def main():
  print('[{}] Build model'.format(task_index))
  model = get_model(batch_size=batch_size)
  step = 0
  print('[{}] Begin Training'.format(task_index))
  while not step < 1000:
    batch_xs, batch_ys = next_batch(batch_size=batch_size)
    loss,deltas = get_deltas(model,batch_xs,batch_ys)
    set_new_weights(model,deltas)
    sys.stdout.write('\rWorker {}, step: {}, loss: {}'.format(task_index,step,loss))
    sys.stdout.flush()


if __name__ == "__main__":
  main() 

