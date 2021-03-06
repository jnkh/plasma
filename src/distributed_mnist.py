from __future__ import print_function
import math,os,sys,time
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
from tensorflow.examples.tutorials.mnist import input_data

NUM_GPUS = 4
IMAGE_PIXELS = 28
hidden_units = 20
batch_size = 2048
sync_mode = True
data_dir = '/tigress/jk7/tmp/data'
from mpi_launch_tensorflow import get_mpi_cluster_server_jobname



def get_loss_accuracy_ops():
  input_tensor = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
  true_output_tensor = tf.placeholder(tf.float32, [None, 10])

  '''How to deal with RNN preserved state between runs?
  Probably need to explicitly feed the last state into the model as the current input state. 
  Or maybe set an explicity variable overwriting the Keras state, and capture it as an explicity output.'''

  x = Dense(hidden_units,activation='relu')(input_tensor)
  x = Dropout(0.1)(x)
  output_tensor = Dense(10,activation='softmax')(x) 


  loss = -tf.reduce_sum(true_output_tensor * tf.log(tf.clip_by_value(output_tensor, 1e-10, 1.0)))
  correct_prediction = tf.equal(tf.argmax(output_tensor, 1), tf.argmax(true_output_tensor, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  return loss,accuracy,input_tensor,true_output_tensor



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

      loss,accuracy,input_tensor,true_output_tensor = get_loss_accuracy_ops()

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

    mnist = input_data.read_data_sets(data_dir, one_hot=True)

    # The supervisor takes care of session initialization, restoring from
    # a checkpoint, and closing when done or an error occurs.
    config = tf.ConfigProto(allow_soft_placement=True)
    with sv.prepare_or_wait_for_session(server.target,config=config) as sess:
      if sync_mode and is_chief:
        sv.start_queue_runners(sess,[chief_queue_runner])
        sess.run(init_tokens_op)

      step = 0
      start = time.time()
      while not sv.should_stop() and step < 1000:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        train_feed = {input_tensor: batch_xs, true_output_tensor: batch_ys,K.learning_phase(): 1}

        _, step, curr_loss, curr_accuracy = sess.run([train_op, global_step, loss, accuracy], feed_dict=train_feed)
      	sys.stdout.write('\rWorker {}, step: {}, loss: {}, accuracy: {}'.format(task_index,step,curr_loss,curr_accuracy))
      	sys.stdout.flush()

    # Ask for all the services to stop.
    sv.stop()
    print('Elapsed: {}'.format(time.time() - start))

if __name__ == "__main__":
  tf.app.run()

