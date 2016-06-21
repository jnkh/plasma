from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy
import tensorflow as tf

tf.app.flags.DEFINE_string("ps_hosts", "localhost:2222", "...")
tf.app.flags.DEFINE_string("worker_hosts", "localhost:2223", "...")
tf.app.flags.DEFINE_string("job_name", "", "...")
tf.app.flags.DEFINE_integer("task_index", 0, "...")
tf.app.flags.DEFINE_integer('gpu_cards', 4, 'Number of GPU cards in a machine to use.')
FLAGS = tf.app.flags.FLAGS

def dense_to_one_hot(labels_dense, num_classes = 10) :
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def run_training(server, cluster_spec, num_workers) :
    is_chief = (FLAGS.task_index == 0)
    with tf.Graph().as_default():        
        with tf.device(tf.train.replica_device_setter(cluster = cluster_spec)) :            
            with tf.device('/cpu:0') :
                global_step = tf.get_variable('global_step', [],
                    initializer = tf.constant_initializer(0), trainable = False)
            with tf.device('/gpu:%d' % (FLAGS.task_index % FLAGS.gpu_cards)) :                            
                # Create the model
                x = tf.placeholder("float", [None, 784])
                W = tf.Variable(tf.zeros([784, 10]))
                b = tf.Variable(tf.zeros([10]))
                y = tf.nn.softmax(tf.matmul(x, W) + b)

                # Define loss and optimizer
                y_ = tf.placeholder("float", [None, 10])
                cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
                opt = tf.train.GradientDescentOptimizer(0.01)
                opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate = num_workers,
                    replica_id = FLAGS.task_index, total_num_replicas = num_workers)
                train_step = opt.minimize(cross_entropy, global_step = global_step)

                # Test trained model
                correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            init_token_op = opt.get_init_tokens_op()
            chief_queue_runner = opt.get_chief_queue_runner()

            init = tf.initialize_all_variables()
            sv = tf.train.Supervisor(is_chief = is_chief,
                init_op = init,
                global_step = global_step)
            # Create a session for running Ops on the Graph.
            config = tf.ConfigProto(allow_soft_placement = True)
            sess = sv.prepare_or_wait_for_session(server.target, config = config)

            if is_chief:
                sv.start_queue_runners(sess, [chief_queue_runner])                
                sess.run(init_token_op)

            for i in range(100000):
                source_data = numpy.random.normal(loc = 0.0, scale = 1.0, size = (100, 784))
                labels_dense = numpy.clip(numpy.sum(source_data, axis = 1) / 5 + 5, 0, 9).astype(int)
                labels_one_hot = dense_to_one_hot(labels_dense)
                _, cost, acc, step = sess.run([train_step, cross_entropy, accuracy, global_step], feed_dict = { x: source_data, y_ : labels_one_hot })
                print("[%d]: cost=%.2f, accuracy=%.2f" % (step, cost, acc))


def main(_) :
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",") 
    num_workers = len(worker_hosts)    
    print("gup_cards=%d; num_worders=%d" % (FLAGS.gpu_cards, num_workers))
    cluster_spec = tf.train.ClusterSpec({ "ps":ps_hosts, "worker" : worker_hosts })    
    server = tf.train.Server(cluster_spec, job_name = FLAGS.job_name, task_index = FLAGS.task_index)
    if FLAGS.job_name == "ps":
      server.join()
    elif FLAGS.job_name == "worker" :
      run_training(server, cluster_spec, num_workers)

if __name__ == '__main__' :
  tf.app.run()




