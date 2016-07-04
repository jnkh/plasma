import tensorflow as tf
import math,os,sys,time
from tensorflow.examples.tutorials.mnist import input_data

# Flags for defining the tf.train.ClusterSpec
# tf.app.flags.DEFINE_string("ps_hosts", "",
#                            "Comma-separated list of hostname:port pairs")
# tf.app.flags.DEFINE_string("worker_hosts", "",
#                            "Comma-separated list of hostname:port pairs")
# # Flags for defining the tf.train.Server
# tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
# tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
# tf.app.flags.DEFINE_integer("hidden_units", 20, "Number of hidden units")
# tf.app.flags.DEFINE_integer("batch_size", 2048, "Batch size")
# tf.app.flags.DEFINE_string("data_dir", '/tigress/jk7/tmp/data', "Data dir")

# FLAGS = tf.app.flags.FLAGS
NUM_GPUS = 4
IMAGE_PIXELS = 28
hidden_units = 20
batch_size = 2048
sync_mode = True
data_dir = '/tigress/jk7/tmp/data'
from mpi_launch_tensorflow import get_mpi_cluster_server_jobname



def get_loss_accuracy_ops():
  # Variables of the hidden layer
  hid_w = tf.Variable(
      tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, hidden_units],
                          stddev=1.0 / IMAGE_PIXELS), name="hid_w")
  hid_b = tf.Variable(tf.zeros([hidden_units]), name="hid_b")

  # Variables of the softmax layer
  sm_w = tf.Variable(
      tf.truncated_normal([hidden_units, 10],
                          stddev=1.0 / math.sqrt(hidden_units)),
      name="sm_w")
  sm_b = tf.Variable(tf.zeros([10]), name="sm_b")

  x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
  y_ = tf.placeholder(tf.float32, [None, 10])

  hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
  hid = tf.nn.relu(hid_lin)

  y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
  loss = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  return loss,accuracy,x,y_



def main(_):
  cluster,server,job_name,task_index,num_workers = get_mpi_cluster_server_jobname(num_ps = 2, num_workers = None)
  MY_GPU = task_index % NUM_GPUS
  # if job_name == "ps":
  #   os.environ['CUDA_VISIBLE_DEVICES'] = ''
  # if job_name == "worker":
  #   os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(MY_GPU)

  if job_name == "ps":
    server.join()
  elif job_name == "worker":

    is_chief = (task_index == 0)
    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(\
      worker_device='/job:worker/task:{}/gpu:{}'.format(task_index,MY_GPU),
		  cluster=cluster)):

      loss,accuracy,x,y_ = get_loss_accuracy_ops()

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
        train_feed = {x: batch_xs, y_: batch_ys}

        _, step, curr_loss, curr_accuracy = sess.run([train_op, global_step,loss,accuracy], feed_dict=train_feed)
      	sys.stdout.write('\rWorker {}, step: {}, loss: {}, accuracy: {}'.format(task_index,step,curr_loss,curr_accuracy))
      	sys.stdout.flush()

    # Ask for all the services to stop.
    sv.stop()
    print('Elapsed: {}'.format(time.time() - start))

if __name__ == "__main__":
  tf.app.run()

