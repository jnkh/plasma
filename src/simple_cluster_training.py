import tensorflow as tf
import numpy as np



def variable_on_device(name,shape,initializer=None,device='/cpu:0'):
	with tf.device(device):
		var = tf.get_variable(name,shape,initializer=initializer)
	return var




def get_weights(num_features):
	W = np.array([0.3,-1.2])#np.random.randn(1,num_features)
	b = np.array([0.5])#np.random.randn(1)
	return W,b

def get_training_examples(batchsize,num_features):
	W,b = get_weights(num_features)
	x = np.random.randn(num_features,batchsize)
	y_mask = (W*x + b) > 0
	y = np.zeros_like(y_mask)
	y[y_mask] = 1
	y[~y_mask] = 0
	return x.T,y





##build model

num_features = 2
# num_hidden = 2
num_out = 2
x = tf.placeholder(tf.float32,[None,num_features])
labels = tf.placeholder(tf.int64, [None,1])

W = variable_on_device('w',[num_features,num_out])
b = variable_on_device('w',[num_out])


with tf.device('/gpu:0'):
	logits = tf.matmul(W,x) + b
	probs = tf.nn.softmax(logits)
	loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits,labels)
	optimizer = tf.train.GradientDescentOptimizer(0.001)
	global_step = tf.Variable(0,name='global_step',trainable=False)
	train_op = optimizer.minimize(loss,global_step=global_step)

init = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)
	for i in xrange(1000):
		x_batch,y_batch = get_training_examples(batchsize,num_features)
		_,loss = sess.run([train_op,loss],feed_dict = {x : x_batch, y : y_batch})

