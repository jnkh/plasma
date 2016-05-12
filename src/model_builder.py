from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.utils.data_utils import get_file

def build_model(conf):
	model_conf = conf['model']
	rnn_size = model_conf['rnn_size']
	rnn_type = model_conf['rnn_type']
	optimizer = model_conf['optimizer']
	dropout_prob = model_conf['dropout_prob']
	length = model_conf['length']
	num_signals = conf['data']['num_signals']


	if rnn_type == 'LSTM':
		rnn_model = LSTM
	elif rnn_type == 'SimpleRNN':
		rnn_model =SimpleRNN 
	else:
		print('Unkown Model Type, exiting.')
		exit(1)

	model = Sequential()
	model.add(rnn_model(rnn_size, return_sequences=False, input_shape=(length, num_signals)))
	model.add(Dropout(dropout_prob))
	model.add(Dense(1))
	model.add(Activation('sigmoid')) #add if probabilistic output
	model.compile(loss='binary_crossentropy', optimizer=optimizer)
	#model.compile(loss='mean_squared_error', optimizer='sgd') #for numerical output
	return model
