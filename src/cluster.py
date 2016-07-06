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
import dill
from functools import partial

#matplotlib
import matplotlib
matplotlib.use('Agg')
import numpy as np


os.environ['KERAS_BACKEND'] = 'tensorflow'

#import keras sequentially because it otherwise reads from ~/.keras/keras.json with too many threads.
from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_task_index = mpi_comm.Get_rank()
mpi_task_num = mpi_comm.Get_size()
for i in range(mpi_task_num):
  mpi_comm.Barrier()
  if i == mpi_task_index:
    from keras import backend as K
    from keras.layers import Dense, Dropout
import tensorflow as tf
import tflearn as tfl


NUM_GPUS = 4
IMAGE_PIXELS = 28
hidden_units = 20
batch_size = 2048
sync_mode = True
data_dir = '/tigress/jk7/tmp/data'
from mpi_launch_tensorflow import get_mpi_cluster_server_jobname



def get_loss_accuracy_ops(batch_size = 32,timesteps = 64, featurelen=1):

    num_layers = 2
    num_hidden = 10
    num_output = 1
    dropout = 0.1

    recurrent_layer = tfl.layers.recurrent.lstm
    state_size = 2*num_hidden
    state_shapes = [(batch_size,state_size) for _ in range(num_layers)]

    #use this when tensorflow doesn't throw an error anymore? Maybe in tensorflow 0.9?
    #initial_states_defaults = [tf.Variable(tf.tile(tf.zeros([1,state_size]),[batch_size,1]),name='trainable initial state {}'.format(i)) for i in range(num_layers)] 
    #initial_states = [tf.placeholder_with_default(initial_states_defaults[i],(batch_size,state_size)) for i in range(num_layers)] 
    initial_states = [tf.placeholder(tf.float32,state_shapes[i]) for i in range(num_layers)] 
    final_states = [None for i in range(num_layers)]

    batch_input_shape = (batch_size,timesteps,featurelen)

    input_tensor = tf.placeholder(tf.float32, batch_input_shape)
    true_output_tensor = tf.placeholder(tf.float32, (batch_size,timesteps,num_output) )


    x = input_tensor
    for layer_index in range(num_layers):
      x,final_states[layer_index] = recurrent_layer(x,num_hidden,dropout = dropout,
      return_seq=True,return_states=True,initial_state=initial_states[layer_index])
    #x is now list of len timesteps with shape (batchsize,num_hidden)
    x = tf.pack(x)
    #x.shape is now (timesteps,batchsize,num_hidden)
    x = tf.transpose(x,[1,0,2])
    #x.shape is now (batchsize,timesteps,num_hidden)
    x = tf.reshape(x,[batch_size*timesteps,num_hidden])
    #x.shape is now (batchsize*timesteps,num_hidden)
    x = tfl.fully_connected(x,num_output,activation = 'tanh')
    #x.shape is now (batchsize,timesteps,num_hidden)
    output_tensor = tf.reshape(x,[batch_size,timesteps,num_output])
    loss = tf.reduce_mean(tfl.losses.L2(output_tensor - true_output_tensor))

    return loss,initial_states,final_states,input_tensor,true_output_tensor,state_shapes
    # x = Dense(hidden_units,activation='relu')(input_tensor)
    # x = Dropout(0.1)(x)
    # output_tensor = Dense(10,activation='softmax')(x) 


    # loss = -tf.reduce_sum(true_output_tensor * tf.log(tf.clip_by_value(output_tensor, 1e-10, 1.0)))
    # correct_prediction = tf.equal(tf.argmax(output_tensor, 1), tf.argmax(true_output_tensor, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # return loss,accuracy,input_tensor,true_output_tensor

def next_batch(batch_size=32,timesteps = 64,featurelen = 1):
  lag = 20
  x = np.random.randn(32,64+lag,1) 
  x = np.cumsum(x,axis=1)
  return x[:,lag:,:],x[:,:-lag,:]


def main(_):
  cluster,server,job_name,task_index,num_workers = get_mpi_cluster_server_jobname(num_ps = 4, num_workers = 5)
  MY_GPU = task_index % NUM_GPUS

  if job_name == "ps":
    server.join()
  elif job_name == "worker":

    is_chief = (task_index == 0)
    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(\
      worker_device='/job:worker/task:{}/gpu:{}'.format(task_index,MY_GPU),
		  cluster=cluster)):

      loss,initial_states,final_states,input_tensor,true_output_tensor,state_shapes = get_loss_accuracy_ops()

      global_step = tf.Variable(0,trainable=False)
      optimizer = tf.train.AdagradOptimizer(0.01)
      if sync_mode:
        optimizer = tf.train.SyncReplicasOptimizer(optimizer,replicas_to_aggregate=num_workers,
          replica_id=task_index,total_num_replicas=num_workers)

      train_op = optimizer.minimize(loss, global_step=global_step)

      if sync_mode and is_chief:
        # Initial token and chief queue runners required by the sync_replicas mode
        chief_queue_runner = optimizer.get_chief_queue_runner()
        init_tokens_op = optimizer.get_init_tokens_op()

      saver = tf.train.Saver()
      summary_op = tf.merge_all_summaries()
      init_op = tf.initialize_all_variables()

    # Create a "supervisor", which oversees the training process.
    sv = tf.train.Supervisor(is_chief=is_chief,logdir="/tmp/train_logs",init_op=init_op,summary_op=summary_op,
                             saver=saver,global_step=global_step,save_model_secs=600)


    # The supervisor takes care of session initialization, restoring from
    # a checkpoint, and closing when done or an error occurs.
    config = tf.ConfigProto(allow_soft_placement=True)
    with sv.prepare_or_wait_for_session(server.target,config=config) as sess:
      if sync_mode and is_chief:
        sv.start_queue_runners(sess,[chief_queue_runner])
        sess.run(init_tokens_op)

      step = 0
      start = time.time()
      curr_final_states = [np.zeros(s_) for s_ in state_shapes]
      while not sv.should_stop() and step < 1000:
        batch_xs, batch_ys = next_batch(batch_size)
        if step == 0:
          train_feed = {input_tensor: batch_xs, true_output_tensor: batch_ys}
        else:
          l = [(input_tensor, batch_xs),(true_output_tensor, batch_ys)] + zip(initial_states,curr_final_states)
          train_feed = { k:v for (k,v) in l}


	print('train_op: {}'.format(train_op))
	print('global_step: {}'.format(global_step))
	print('loss: {}'.format(loss))
	print('final_states: {}'.format(final_states))
	print('train_feed: {}'.format(train_feed))
        outs = sess.run([train_op, global_step, loss] +  final_states, feed_dict=train_feed)
        step = outs[1]
        curr_loss = outs[2]
        curr_final_states = outs[3:]
      	sys.stdout.write('\rWorker {}, step: {}, loss: {}'.format(task_index,step,curr_loss))
      	sys.stdout.flush()

    # Ask for all the services to stop.
    sv.stop()
    print('Elapsed: {}'.format(time.time() - start))

if __name__ == "__main__":
  tf.app.run()

