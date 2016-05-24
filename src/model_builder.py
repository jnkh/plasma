from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.utils.data_utils import get_file
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import Callback
from keras.optimizers import *

import dill
import re,os

from data_processing import Loader

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


class ModelBuilder():
	def __init__(self,conf):
		self.conf = conf

	def get_unique_id(self):
		this_conf = self.conf.copy()
		#don't make hash dependent on number of epochs.
		this_conf['training']['num_epochs'] = 0
		return hash(dill.dumps(this_conf))


	def build_model(self,predict):
		conf = self.conf
		model_conf = conf['model']
		rnn_size = model_conf['rnn_size']
		rnn_type = model_conf['rnn_type']
		optimizer = model_conf['optimizer']
		if optimizer == 'sgd':
			optimizer = SGD(lr=0.0001)
		if optimizer == 'adam':
			optimizer = Adam(lr=0.0001)
		loss_fn = model_conf['loss']
		dropout_prob = model_conf['dropout_prob']
		length = model_conf['length']
		pred_length = model_conf['pred_length']
		skip = model_conf['skip']
		stateful = model_conf['stateful']
		return_sequences = model_conf['return_sequences']
		num_signals = conf['data']['num_signals']

		batch_size = Loader.get_batch_size(self.conf['training']['batch_size'],predict)
		if predict:
		    #so we can predict with one time point at a time!
			length =pred_length 


		if rnn_type == 'LSTM':
			rnn_model = LSTM
		elif rnn_type == 'SimpleRNN':
			rnn_model =SimpleRNN 
		else:
			print('Unkown Model Type, exiting.')
			exit(1)

		model = Sequential()
		model.add(rnn_model(rnn_size, return_sequences=return_sequences,
		 stateful=stateful, batch_input_shape=(batch_size,length, num_signals)))
		model.add(Dropout(dropout_prob))
		if return_sequences:
			model.add(TimeDistributed(Dense(1)))
			model.add(TimeDistributed(Activation('sigmoid'))) #add if probabilistic output
		else:
			model.add(Dense(1))
			model.add(Activation('sigmoid')) #add if probabilistic output
		model.compile(loss=loss_fn, optimizer=optimizer)
		model.reset_states()
		#model.compile(loss='mean_squared_error', optimizer='sgd') #for numerical output
		return model

	def build_train_test_models(self):
		return self.build_model(False),self.build_model(True)

	def save_model_weights(self,model,epoch):
		save_path = self.get_save_path(epoch)
		model.save_weights(save_path,overwrite=True)

	def get_save_path(self,epoch):
		unique_id = self.get_unique_id()
		return self.conf['paths']['model_save_path'] + 'model.{}._epoch_.{}.h5'.format(unique_id,epoch)


	def load_model_weights(self,model):
		unique_id = self.get_unique_id()
		epochs = self.get_all_saved_files()
		if len(epochs) == 0:
			print('no previous checkpoint found')
			return -1
		else:
			max_epoch = max(epochs)
			print('loading from epoch {}'.format(max_epoch))
			model.load_weights(self.get_save_path(max_epoch))
			return max_epoch

	def extract_id_and_epoch_from_filename(self,filename):
		regex = re.compile(r'-?\d+')
		numbers = [int(x) for x in regex.findall(filename)]
		assert(len(numbers) == 3) #id,epoch number and extension
		assert(numbers[2] == 5) #.h5 extension
		return numbers[0],numbers[1]


	def get_all_saved_files(self):
		unique_id = self.get_unique_id()
		filenames = os.listdir(self.conf['paths']['model_save_path'])
		epochs = []
		for file in filenames:
			curr_id,epoch = self.extract_id_and_epoch_from_filename(file)
			if curr_id == unique_id:
				epochs.append(epoch)
		return epochs




