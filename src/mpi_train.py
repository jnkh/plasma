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


hidden_units = 20
batch_size = 512
sync_mode = True
lr = 0.005
data_dir = '/tigress/jk7/tmp/data'


def get_model(batch_size = 32,timesteps = 100, featurelen=1,is_training=True,lr = 0.001):

    num_layers = 2
    num_hidden = 10
    num_output = 1
    dropout = 0.1

    input_tensor = Input(batch_shape=(batch_size,timesteps,featurelen))
    recurrent_layer = LSTM(hidden_units,return_sequences=True,stateful = True)(input_tensor)
    output_tensor = TimeDistributed(Dense(num_output,activation='linear'))(recurrent_layer)

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


def batch_iterator(batch_size=32,timesteps = 100,featurelen = 1):
  multiplier = 100
  lag = 200
  while True:
    xx = np.random.randn(batch_size,multiplier*timesteps+lag,featurelen) 
    xx = np.cumsum(xx,axis=1)
    xx = xx/np.max(np.abs(xx))
    for chunk_idx in xrange(multiplier):
      start = chunk_idx*timesteps
      stop = (1+chunk_idx)*timesteps
      x_batch = xx[:,start+lag:stop+lag,:]
      y_batch = xx[:,start:stop,:]
      yield x_batch,y_batch


def mpi_reduce_array(arr):
  arr_global = np.empty_like(arr)
  comm.Allreduce(arr,arr_global,op=MPI.SUM)
  arr_global /= num_workers
  return arr_global

def get_deltas(model,X_batch,Y_batch,verbose=False):
  if task_index == 0 and verbose: time0 = time.time()
  weights_before_update = model.get_weights()
  if task_index == 0 and verbose: time1 = time.time()
  loss = model.train_on_batch(X_batch,Y_batch)
  if task_index == 0 and verbose: time2 = time.time()
  weights_after_update = model.get_weights()
  if task_index == 0 and verbose: time3 = time.time()
  deltas = [w1 - w0 for w1,w0 in zip(weights_after_update,weights_before_update)]
  if task_index == 0 and verbose: time4 = time.time()
  if task_index == 0 and verbose:
    sys.stdout.write('\ntime 0: {}'.format(time1 - time0))
    sys.stdout.write('\ntime 1: {}'.format(time2 - time1))
    sys.stdout.write('\ntime 2: {}'.format(time3 - time2))
    sys.stdout.write('\ntime 3: {}'.format(time4 - time3))
    sys.stdout.flush()
  return deltas,loss


def apply_deltas(model,deltas):
  model.set_weights()

def get_new_weights(model,deltas):
  return [w-d for w,d in zip(model.get_weights(),deltas)]

def sync_deltas(deltas,single_worker=False):
  global_deltas = []
  #default is to reduce the deltas from all workers
  if not single_worker:
    for delta in deltas:
      global_deltas.append(mpi_reduce_array(delta))
  #otherwise keep only the local deltas
  else:
    global_deltas = deltas
  return global_deltas 

def set_new_weights(model,deltas,single_worker=False):
  # if single_worker is True, only the rank 0 deltas are used. Otherwise all of them are.
  global_deltas = sync_deltas(deltas,single_worker)
  if comm.rank == 0:
    new_weights = get_new_weights(model,global_deltas)
  else:
    new_weights = None
  new_weights = comm.bcast(new_weights,root=0)
  model.set_weights(new_weights)



def train_epoch(model,batch_size=32,train_steps=100,warmup_steps=100):
  verbose = False
  step = 0
  print('[{}] Begin Training'.format(task_index))
  for batch_xs,batch_ys in batch_iterator(batch_size=batch_size):
    if step >= train_steps:
     break
    warmup_phase = step < warmup_steps

    if task_index == 0 and verbose:
      start_time = time.time()

    if task_index == 0 and verbose:
      batch_time = time.time()

    deltas,loss = get_deltas(model,batch_xs,batch_ys,verbose)

    if task_index == 0 and verbose:
      deltas_time = time.time()

    set_new_weights(model,deltas,warmup_phase)

    if task_index == 0 and verbose:
      sync_time = time.time()
    if task_index == 0 and verbose:
      sys.stdout.write('\nget deltas: {:.3f}'.format(deltas_time - batch_time))
      sys.stdout.write('\nproduce batch: {:.3f}'.format(batch_time - start_time))
      sys.stdout.write('\nsync deltas: {:.3f}\n'.format(sync_time - deltas_time))
      sys.stdout.flush()
    write_str = '\rWorker {}, step: {}, loss: {}'.format(task_index,step,loss)
    if warmup_phase:
      write_str += ' [Warmup]'
    sys.stdout.write(write_str)
    sys.stdout.flush()
    step += 1
  return model


def test(model,batch_size=1):
  ys_pred_list = []
  xs_list = []
  ys_true_list = []
  num_concat = 200
  for i,(batch_xs,batch_ys) in enumerate(batch_iterator(batch_size=batch_size)):
    if i >= num_concat:
      break

    ys_pred_list.append(model.predict(batch_xs,batch_size=batch_size))
    xs_list.append(batch_xs)
    ys_true_list.append(batch_ys)
  ys_pred = np.squeeze(np.concatenate(ys_pred_list,axis=1))
  xs = np.squeeze(np.concatenate(xs_list,axis=1))
  ys_true = np.squeeze(np.concatenate(ys_true_list,axis=1))
  print('Testing loss: {}'.format(np.mean((ys_pred - ys_true)**2)))

  plt.plot(xs,'b')
  plt.plot(ys_pred,'r')
  plt.plot(ys_true,'g')
  plt.show()
  plt.savefig('out_{}.png'.format(time.time()),bbox_inches='tight')



def main():
  save_path = 'tmp_mpi/model_weights_epoch{}.h5'#{}.h5'
  warmup_steps = 100
  train_steps = 100
  epochs = 10
  print('[{}] Build model'.format(task_index))
  for e in range(epochs):
    warmup_steps_curr = warmup_steps if e == 0 else 0
    model = get_model(batch_size=batch_size,lr=lr)
    model = train_epoch(model,batch_size,train_steps = train_steps,warmup_steps=warmup_steps_curr)

    if task_index == 0:
      print('Evaluating model...')
      save_path_curr = save_path.format(e)
      model.save_weights(save_path_curr)
      test_model = get_model(batch_size = 1)
      test_model.load_weights(save_path_curr)
      test(test_model)
      print('done.')






if __name__ == "__main__":
  main() 

