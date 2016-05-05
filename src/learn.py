import matplotlib
matplotlib.use('Agg')

from data_processing import *


############Script###########

signal_prepath = '../data/signal_data/jet/';

signals_dirs = ['jpf/da/c2-ipla','jpf/da/c2-loca','jpf/db/b5r-ptot>out',
                'jpf/df/g1r-lid:003','jpf/gs/bl-li<s','jpf/gs/bl-fdwdt<s',
                'jpf/gs/bl-ptot<s','jpf/gs/bl-wmhd<s']
num_signals = len(signals_dirs)

shots_and_times_path = '../data/shot_lists/short_list_times_cf.txt'

#train/validate split
train_frac = 0.8

#how many shots to use
use_shots = 5

#normalization timescale
dt = 0.001

#maximum TTD considered
T_max = 100

#length of LSTM memory
length = 50
skip = 1

#training params
batch_size = 200
num_epochs = 10



print("Read in data")
#get shot information from preprocessed files
shots,min_times,max_times = get_shots_and_times(shots_and_times_path)

#read signals from data files
signals,ttd = get_signals_and_ttds(signal_prepath,signals_dirs,shots,min_times,max_times,T_max,dt,use_shots)
signals_train,signals_test = train_test_split(signals,train_frac)
ttd_train,ttd_test = train_test_split(ttd,train_frac)

#convert to usable training data format
X,y = array_to_path_and_external_pred(signals,ttd,length,skip)
X_train,y_train = array_to_path_and_external_pred(signals_train,ttd_train,length,skip)
X_test,y_test = array_to_path_and_external_pred(signals_test,ttd_test,length,skip)
print("...done")


print('Build model...')
model = Sequential()
model.add(SimpleRNN(20, return_sequences=False, input_shape=(length, num_signals)))
model.add(Dropout(0.05))
model.add(Dense(1))
#model.add(Activation('tanh'))
model.compile(loss='mean_squared_error', optimizer='sgd')
print('...done')


print('training model')
model.fit(X_train,y_train,batch_size=batch_size,nb_epoch=num_epochs,verbose=1,validation_split=0.0)
print('...done')

print('evaluating model')
res = model.evaluate(X_test,y_test)
print(res)


print('plotting results')
ttd_prime = model.predict(X)
ttd_prime_test = model.predict(X_test)
ttd_prime_train = model.predict(X_train)

plot(ttd)
indices_train = range(length,len(y_train)+length)
indices_test = range(len(y_train)+length,len(y)+1)
plot(ttd_prime)
plot(indices_test,ttd_prime_test,'g')
plot(indices_train,ttd_prime_train,'r')
savefig('plot.png')
#plot(y_train,'.')
#show()

