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






class MPIModel():
  def __init__(self,model,comm,batch_iterator,batch_size,num_replicas=None,warmup_steps=1000,lr=0.01):
    self.epoch = 0
    self.model = model
    self.lr = lr
    self.DUMMY_LR = 0.1
    self.max_lr = 0.5
    self.comm = comm
    self.batch_size = batch_size
    self.batch_iterator = batch_iterator
    self.warmup_steps=warmup_steps
    self.num_workers = comm.Get_size()
    self.task_index = comm.Get_rank()
    if num_replicas is None or num_replicas < 1 or num_replicas > num_workers:
      self.num_replicas = num_workers
    else:
      self.num_replicas = num_replicas


  def set_lr(self,lr):
    self.lr = lr

  def save_weights(self,path,overwrite=False):
    self.model.save_weights(path,overwrite=overwrite)

  def load_weights(self,path):
    self.model.load_weights(path)

  def compile(self,loss='mse'):
    self.model.compile(optimizer=SGD(lr=self.DUMMY_LR),loss=loss)


  def get_deltas(self,X_batch,Y_batch,verbose=False):
    weights_before_update = self.model.get_weights()

    loss = self.model.train_on_batch(X_batch,Y_batch)

    weights_after_update = self.model.get_weights()
    self.model.set_weights(weights_before_update)

    deltas = subtract_params(weights_after_update,weights_before_update)
    deltas = multiply_params(deltas,1.0/self.DUMMY_LR)

    return deltas,loss


  def get_new_weights(self,deltas):
    return add_params(self.model.get_weights(),deltas)

  def mpi_average_gradients(self,arr,num_replicas=None):
    if num_replicas == None:
      num_replicas = self.num_workers 
    if self.task_index >= num_replicas:
      arr *= 0.0
    arr_global = np.empty_like(arr)
    self.comm.Allreduce(arr,arr_global,op=MPI.SUM)
    arr_global /= num_replicas
    return arr_global



  def mpi_average_scalars(self,val,num_replicas=None):
    if num_replicas == None:
      num_replicas = self.num_workers 
    if self.task_index >= num_replicas:
      val *= 0.0
    val_global = 0.0 
    val_global = self.comm.allreduce(val,op=MPI.SUM)
    val_global /= num_replicas
    return val_global 


  def sync_deltas(self,deltas,num_replicas=None):
    global_deltas = []
    #default is to reduce the deltas from all workers
    for delta in deltas:
      global_deltas.append(self.mpi_average_gradients(delta,num_replicas))
    return global_deltas 

  def set_new_weights(self,deltas,num_replicas=None):
    global_deltas = self.sync_deltas(deltas,num_replicas)
    effective_lr = self.get_effective_lr(num_replicas)


    global_deltas = multiply_params(global_deltas,effective_lr)
    if comm.rank == 0:
      new_weights = self.get_new_weights(global_deltas)
    else:
      new_weights = None
    new_weights = self.comm.bcast(new_weights,root=0)
    self.model.set_weights(new_weights)



  def train_epoch(self):
    verbose = False
    step = 0
    for batch_xs,batch_ys,reset_states_now,epoch_end in self.batch_iterator():
      if reset_states_now:
        self.model.reset_states()

      warmup_phase = (step < self.warmup_steps and self.epoch == 0)
      num_replicas = 1 if warmup_phase else self.num_replicas

      t0 = time.time()
      deltas,loss = self.get_deltas(batch_xs,batch_ys,verbose)
      t1 = time.time()
      self.set_new_weights(deltas,num_replicas)
      t2 = time.time()
      write_str_0 = self.calculate_speed(t0,t1,t2,num_replicas)


      write_str = '\r[{}] step: {}, loss: {:.2E} | '.format(self.task_index,step,self.mpi_average_scalars(1.0*loss,num_replicas))
      print_unique(write_str + write_str_0)
      step += 1
      if epoch_end:
        self.epoch += 1
        break


  def get_effective_lr(self,num_replicas):
    effective_lr = self.lr * num_replicas
    if effective_lr > self.max_lr:
      print_unique('Warning: effective learning rate set to {}, larger than maximum {}. Clipping.'.format(effective_lr,self.max_lr))
      effective_lr = self.max_lr
    return effective_lr

  def get_effective_batch_size(self,num_replicas):
    return self.batch_size*num_replicas

  def train_epochs(self,warmup_steps=100,num_epochs=1):
    for i in range(num_epochs):
      self.train_epoch(warmup_steps)


  def calculate_speed(self,t0,t_after_deltas,t_after_update,num_replicas,verbose=False):
    effective_batch_size = self.get_effective_batch_size(num_replicas)
    t_calculate = t_after_deltas - t0
    t_sync = t_after_update - t_after_deltas
    t_tot = t_after_update - t0

    examples_per_sec = effective_batch_size/t_tot
    frac_calculate = t_calculate/t_tot
    frac_sync = t_sync/t_tot

    print_str = '{:.2E} Examples/sec | {:.2E} sec/batch [{:.1%} calc., {:.1%} synch.]'.format(examples_per_sec,t_tot,frac_calculate,frac_sync)
    print_str += '[batch = {} = {}*{}]'.format(effective_batch_size,self.batch_size,num_replicas)
    if verbose:
      print_unique(print_str)
    return print_str






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
        reset_states_now = chunk_idx == 0
        start = chunk_idx*timesteps
        stop = (1+chunk_idx)*timesteps
        x_batch = xx[:,start:stop,:]
        y_batch = yy[:,start:stop,:]
        global_step += 1
        yield x_batch,y_batch,reset_states_now,epoch_end


    if mode == 2:
      xx = 1.0/np.sqrt(multiplier*timesteps)*np.random.randn(batch_size,multiplier*timesteps+lag,featurelen) 
      xx = np.cumsum(xx,axis=1)
      #xx = xx/np.max(np.abs(xx))
      for i in xrange(batch_size):
        xx[i,:,:] = np.roll(xx[i,:,:],np.random.randint(0,multiplier*timesteps+lag),axis=0)
      for chunk_idx in xrange(multiplier):
        epoch_end = global_step == epoch_length - 1
        reset_states_now = chunk_idx == 0
        start = chunk_idx*timesteps
        stop = (1+chunk_idx)*timesteps
        x_batch = xx[:,start+lag:stop+lag,:]
        y_batch = xx[:,start:stop,:]
        global_step += 1
        yield x_batch,y_batch,reset_states_now,epoch_end


def turn_array_into_switch(arr):
  out_arr = np.zeros_like(arr)
  current = 0.0
  for i in range(len(arr)):
    if arr[i] > 0.5:
      current = 1.0 - current
    out_arr[i] = current
  return out_arr





def multiply_params(params,eps):
  return [el*eps for el in params]

def subtract_params(params1,params2):
  return [p1 - p2 for p1,p2 in zip(params1,params2)]

def add_params(params1,params2):
  return [p1 + p2 for p1,p2 in zip(params1,params2)]


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



def print_unique(print_str):
  if task_index == 0:
    sys.stdout.write(print_str)
    sys.stdout.flush()

def print_all(print_str):
  sys.stdout.write('[{}] '.format(task_index) + print_str)
  sys.stdout.flush()




def main():
  save_path = 'tmp_mpi/model_weights_epoch{}.h5'#{}.h5'
  warmup_steps = 500
  train_steps = 500
  epochs = 20
  lr = 0.01
  lr_decay = 1.0
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


