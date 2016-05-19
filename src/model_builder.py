from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.utils.data_utils import get_file
from keras.layers.wrappers import TimeDistributed


def build_model(conf,predict):
	model_conf = conf['model']
	rnn_size = model_conf['rnn_size']
	rnn_type = model_conf['rnn_type']
	optimizer = model_conf['optimizer']
	dropout_prob = model_conf['dropout_prob']
	length = model_conf['length']
	num_signals = conf['data']['num_signals']
	if predict:
		length = 1
    #so we can predict with one time point at a time!


	if rnn_type == 'LSTM':
		rnn_model = LSTM
	elif rnn_type == 'SimpleRNN':
		rnn_model =SimpleRNN 
	else:
		print('Unkown Model Type, exiting.')
		exit(1)

	model = Sequential()
	model.add(rnn_model(rnn_size, return_sequences=True, batch_input_shape=(length,length, num_signals)))
	model.add(Dropout(dropout_prob))
	model.add(TimeDistributed(Dense(1)))
	model.add(Activation('sigmoid')) #add if probabilistic output
	model.compile(loss='binary_crossentropy', optimizer=optimizer)
	#model.compile(loss='mean_squared_error', optimizer='sgd') #for numerical output
	return model
