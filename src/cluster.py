import tensorflow as tf
import math,os,sys,time
from tensorflow.examples.tutorials.mnist import input_data

# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("hidden_units", 20, "Number of hidden units")
tf.app.flags.DEFINE_integer("batch_size", 2048, "Batch size")
tf.app.flags.DEFINE_string("data_dir", '/tigress/jk7/tmp/data', "Data dir")

FLAGS = tf.app.flags.FLAGS
IMAGE_PIXELS = 28
NUM_GPUS = 4
MY_GPU = FLAGS.task_index % NUM_GPUS
#if FLAGS.job_name == "worker":
#  os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(MY_GPU)
from mpi_launch_tensorflow import get_mpi_cluster_server_jobname


def main(_):
  cluster,server,jobname = get_mpi_cluster_server_jobname()
  sys.stdout.write('***')
  sys.stdout.flush()
  if FLAGS.job_name == "ps":
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
  #ps_hosts = FLAGS.ps_hosts.split(",")
  #worker_hosts = FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  #cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  #server = tf.train.Server(cluster,
    #                       job_name=FLAGS.job_name,
    #                       task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(\
      worker_device='/job:worker/task:{}/gpu:{}'.format(FLAGS.task_index,MY_GPU),
		  cluster=cluster)):



      # Variables of the hidden layer
      hid_w = tf.Variable(
          tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
                              stddev=1.0 / IMAGE_PIXELS), name="hid_w")
      hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b")

      # Variables of the softmax layer
      sm_w = tf.Variable(
          tf.truncated_normal([FLAGS.hidden_units, 10],
                              stddev=1.0 / math.sqrt(FLAGS.hidden_units)),
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

      global_step = tf.Variable(0,trainable=False)

      train_op = tf.train.AdagradOptimizer(0.01).minimize(
          loss, global_step=global_step)

      saver = tf.train.Saver()
      summary_op = tf.merge_all_summaries()
      init_op = tf.initialize_all_variables()

    # Create a "supervisor", which oversees the training process.
    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             logdir="/tmp/train_logs",
                             init_op=init_op,
                             summary_op=summary_op,
                             saver=saver,
                             global_step=global_step,
                             save_model_secs=600)

    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    

    # The supervisor takes care of session initialization, restoring from
    # a checkpoint, and closing when done or an error occurs.
    config = tf.ConfigProto(allow_soft_placement=True)
    with sv.prepare_or_wait_for_session(server.target,config=config) as sess:
      # Loop until the supervisor shuts down or 1000000 steps have completed.
      step = 0
      start = time.time()
      while not sv.should_stop() and step < 10000:
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.

        batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
        train_feed = {x: batch_xs, y_: batch_ys}

        _, step, curr_loss, curr_accuracy = sess.run([train_op, global_step,loss,accuracy], feed_dict=train_feed)
      	sys.stdout.write('\rWorker {}, step: {}, loss: {}, accuracy: {}'.format(FLAGS.task_index,step,curr_loss,curr_accuracy))
      	sys.stdout.flush()

    # Ask for all the services to stop.
    sv.stop()
    print('Elapsed: {}'.format(time.time() - start))

if __name__ == "__main__":
  tf.app.run()

