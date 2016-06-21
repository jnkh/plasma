import tensorflow as tf





##build model

num_features = 2
# num_hidden = 2
num_out = 2
x = tf.placeholder(tf.float32,[None,num_features])
labels = tf.placeholder(tf.int64, [None,1])

W = variable_on_device('w',[num_features,num_out])
b = variable_on_device('w',[num_out])


with tf.device('/gpu:0')
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
		_,loss = sess.run([train_op,loss],feed_dict = {x : 
													   y :
													   }



def variable_on_device(name,shape,initializer=None,device='/cpu:0'):
	with tf.device(device):
		var = tf.get_variable(name,shape,initializer=initializer)
	return var