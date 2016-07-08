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


hidden_units = 100
batch_size = 512
sync_mode = True
LR = 0.0001
DUMMY_LR = 0.1
data_dir = '/tigress/jk7/tmp/data'


def get_model(batch_size = 32,timesteps = 100, featurelen=1,is_training=True):

    num_layers = 2
    num_output = 1
    dropout = 0.1

    input_tensor = Input(batch_shape=(batch_size,timesteps,featurelen))
    recurrent_layer = LSTM(hidden_units,return_sequences=True,stateful = True)(input_tensor)
    output_tensor = TimeDistributed(Dense(num_output,activation='linear'))(recurrent_layer)

    model = Model(input =input_tensor,output=output_tensor)
    model.compile(optimizer=SGD(lr=DUMMY_LR),loss='mse')

    return model




def batch_iterator(batch_size=32,timesteps = 10,featurelen = 1):
  multiplier = 1000
  lag = 70
  density = 0.005
  mode = 2
  batch_shape = (batch_size,multiplier*timesteps,featurelen)
  while True:
    if mode == 1:
      xx = np.random.binomial(1,density,batch_shape)
      yy = 1.0*xx
      for i in xrange(batch_size):
        yy[i,:,0] = turn_array_into_switch(xx[i,:,0])
      yy = np.roll(yy,lag)
      for chunk_idx in xrange(multiplier):
        start = chunk_idx*timesteps
        stop = (1+chunk_idx)*timesteps
        x_batch = xx[:,start:stop,:]
        y_batch = yy[:,start:stop,:]
        yield x_batch,y_batch


    if mode == 2:
      xx = np.random.randn(batch_size,multiplier*timesteps+lag,featurelen) 
      xx = np.cumsum(xx,axis=1)
      xx = xx/np.max(np.abs(xx))
      for i in xrange(batch_size):
        xx[i,:,:] = np.roll(xx[i,:,:],np.random.randint(0,multiplier*timesteps+lag),axis=0)
      for chunk_idx in xrange(multiplier):
        start = chunk_idx*timesteps
        stop = (1+chunk_idx)*timesteps
        x_batch = xx[:,start+lag:stop+lag,:]
        y_batch = xx[:,start:stop,:]
        yield x_batch,y_batch


def turn_array_into_switch(arr):
  out_arr = np.zeros_like(arr)
  current = 0.0
  for i in range(len(arr)):
    if arr[i] > 0.5:
      current = 1.0 - current
    out_arr[i] = current
  return out_arr



def get_deltas(model,X_batch,Y_batch,verbose=False):
  weights_before_update = model.get_weights()

  loss = model.train_on_batch(X_batch,Y_batch)

  weights_after_update = model.get_weights()
  model.set_weights(weights_before_update)

  deltas = subtract_params(weights_after_update,weights_before_update)
  deltas = multiply_params(deltas,1.0/DUMMY_LR)

  return deltas,loss


def get_new_weights(model,deltas):
  return add_params(model.get_weights(),deltas)





def multiply_params(params,eps):
  return [el*eps for el in params]

def subtract_params(params1,params2):
  return [p1 - p2 for p1,p2 in zip(params1,params2)]

def add_params(params1,params2):
  return [p1 + p2 for p1,p2 in zip(params1,params2)]


def mpi_average_gradients(arr,num_replicas=None):
  if num_replicas == None:
    num_replicas = num_workers 
  if task_index >= num_replicas:
    arr *= 0.0
  arr_global = np.empty_like(arr)
  comm.Allreduce(arr,arr_global,op=MPI.SUM)
  arr_global /= num_replicas
  return arr_global



def mpi_average_scalars(val,num_replicas=None):
  if num_replicas == None:
    num_replicas = num_workers 
  if task_index >= num_replicas:
    val *= 0.0
  val_global = 0.0 
  val_global = comm.allreduce(val,op=MPI.SUM)
  val_global /= num_replicas
  return val_global 


def sync_deltas(deltas,num_replicas=None):
  global_deltas = []
  #default is to reduce the deltas from all workers
  for delta in deltas:
    global_deltas.append(mpi_average_gradients(delta,num_replicas))
  return global_deltas 

def set_new_weights(model,deltas,num_replicas=None):
  #
  global_deltas = sync_deltas(deltas,num_replicas)
  global_deltas = multiply_params(global_deltas,LR)
  if comm.rank == 0:
    new_weights = get_new_weights(model,global_deltas)
  else:
    new_weights = None
  new_weights = comm.bcast(new_weights,root=0)
  model.set_weights(new_weights)



def train_epoch(model,batch_size=32,train_steps=100,warmup_steps=100):
  verbose = False
  step = 0
  for batch_xs,batch_ys in batch_iterator(batch_size=batch_size):
    if step >= train_steps:
     break

    warmup_phase = step < warmup_steps
    num_replicas = 1 if warmup_phase else num_workers

    deltas,loss = get_deltas(model,batch_xs,batch_ys,verbose)

    set_new_weights(model,deltas,num_replicas)


    write_str = '\r[{}] step: {}, loss: {:.7f}'.format(task_index,step,mpi_average_scalars(1.0*loss,num_replicas))
    write_str += ' [num_replicas = {}]'.format(num_replicas)
    print_unique(write_str)
    step += 1
  return model


def test(model,batch_size=1,epoch=None):
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



def print_unique(print_str):
  if task_index == 0:
    sys.stdout.write(print_str)
    sys.stdout.flush()

def print_all(print_str):
  sys.stdout.write('[{}] '.format(task_index) + print_str)
  sys.stdout.flush()




def main():
  save_path = 'tmp_mpi/model_weights_epoch{}.h5'#{}.h5'
  warmup_steps = 1000
  train_steps = 1000
  epochs = 10
  print_all('Building model\n')
  model = get_model(batch_size=batch_size,timesteps=10)
  for e in range(epochs):
    print_unique('Epoch {}\n'.format(e))
    warmup_steps_curr = warmup_steps if e == 0 else 0


    if task_index == 0:
      print('Evaluating model...')
      save_path_curr = save_path.format(e)
      model.save_weights(save_path_curr,overwrite=True)
      test_model = get_model(batch_size = 1,timesteps=10)
      test_model.load_weights(save_path_curr)
      test(test_model,epoch=e)
      print('done.')



    model = train_epoch(model,batch_size,train_steps = train_steps,warmup_steps=warmup_steps_curr)




if __name__ == "__main__":
  main() 

