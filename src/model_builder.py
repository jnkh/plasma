from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.utils.data_utils import get_file

def build_model(rnn_size,dropout_prob,length,num_signals):
	model = Sequential()
	model.add(SimpleRNN(rnn_size, return_sequences=False, input_shape=(length, num_signals)))
	model.add(Dropout(dropout_prob))
	model.add(Dense(1))
	model.add(Activation('sigmoid')) #add if probabilistic output
	model.compile(loss='binary_crossentropy', optimizer='sgd')
	#model.compile(loss='mean_squared_error', optimizer='sgd') #for numerical output
	return model