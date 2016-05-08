import matplotlib
matplotlib.use('Agg')

from data_processing import *
from model_builder import build_model

#paths#
signal_prepath = '../data/signal_data/jet/';
signals_dirs = ['jpf/da/c2-ipla','jpf/da/c2-loca','jpf/db/b5r-ptot>out',
                'jpf/df/g1r-lid:003','jpf/gs/bl-li<s','jpf/gs/bl-fdwdt<s',
                'jpf/gs/bl-ptot<s','jpf/gs/bl-wmhd<s']
num_signals = len(signals_dirs)
current_index = 0
#shots_and_times_path = '../data/shot_lists/short_list_times_cf.txt'
shots_and_disruption_times_path = '../data/shot_lists/long_list_C.txt' #'short_list.txt' 'long_list_C.txt'
shots_and_minmax_times_path = '../data/shot_lists/long_list_C_minmax_times.txt' #short_list_minmax_times.txt' 'long_list_C_minmax_times.txt'
read_minmax_from_file = True
#processed data
processed_prepath = '../data/processed_shots/'
recompute = False



plotting = False

#train/validate split
as_array_of_shots=True
train_frac = 0.85
shuffle_training = True

#how many shots to use
use_shots = 40

#normalization timescale
dt = 0.001

#maximum TTD considered
T_max = 2
T_warning = 0.1

#length of LSTM memory
length = 100
skip = 1

rnn_size = 20
dropout_prob = 0.1

#training params
batch_size = 2048
num_epochs = 10



print("Generating minmax times")
#get shot information from preprocessed files
if not read_minmax_from_file:
    shots,min_times,max_times = get_shots_and_minmax_times(signal_prepath,signals_dirs,shots_and_disruption_times_path,
               current_index,use_shots,True,shots_and_minmax_times_path)
else:
    shots,min_times,max_times = read_shots_and_minmax_times_from_file(shots_and_minmax_times_path)

print("Reading and cutting signal data")
#read signals from data files
signals_by_shot,ttd_by_shot = get_signals_and_ttds(signal_prepath,signals_dirs,processed_prepath,
    shots,min_times,max_times,T_max,dt,use_shots,recompute,as_array_of_shots)

#ttd remapping: binary -- are we less than thresh away from disruption?


def remap_target(ttd,as_array_of_shots=True):
    binary_ttd = 0*ttd
    mask = ttd < log10(T_warning)
    binary_ttd[mask] = 1.0
    binary_ttd[~mask] = 0.0
    return binary_ttd

if as_array_of_shots:
    ttd_by_shot = array([remap_target(_t) for _t in ttd_by_shot])
else:
    ttd_by_shot = remap_target(ttd_by_shot)


split_groups = train_test_split_all((signals_by_shot,ttd_by_shot),train_frac,shuffle_training)
signals_train_by_shot,signals_test_by_shot = split_groups[0]
ttd_train_by_shot,ttd_test_by_shot = split_groups[1]


num_shots = len(ttd_by_shot)
num_shots_train = len(ttd_train_by_shot)
num_shots_test = len(ttd_test_by_shot)

print("Converting to training data format")
#convert to usable training data format
X_by_shot,y_by_shot = \
    zip(*[array_to_path_and_external_pred(signals_by_shot[i],ttd_by_shot[i],length,skip) for i in range(num_shots)])
X_train_by_shot,y_train_by_shot = \
    zip(*[array_to_path_and_external_pred(signals_train_by_shot[i],ttd_train_by_shot[i],length,skip) for i in range(num_shots_train)])
X_test_by_shot,y_test_by_shot = \
    zip(*[array_to_path_and_external_pred(signals_test_by_shot[i],ttd_test_by_shot[i],length,skip) for i in range(num_shots_test)])
print("...done")


print('Build model...')
model = build_model(rnn_size,dropout_prob,length,num_signals)
print('...done')


print('training model')
for e in range(num_epochs):
    print('Epoch {}/{}'.format(e+1,num_epochs))
    for shot_idx in range(num_shots_train):
        print('Shot {}/{}'.format(shot_idx,num_shots_train))
        model.fit(X_train_by_shot[shot_idx],y_train_by_shot[shot_idx],batch_size=batch_size,nb_epoch=1,verbose=1,validation_split=0.0)
print('...done')

print('evaluating model')
for shot_idx in range(num_shots_test):
    print('Shot {}/{}'.format(shot_idx,num_shots_test))
    res = model.evaluate(X_test_by_shot[shot_idx],y_test_by_shot[shot_idx])
    print(res)


print('saving results')
ttd_prime = [model.predict(_X) for _X in X_by_shot]
ttd_prime_test = [model.predict(_X) for _X in X_test_by_shot]
ttd_prime_train = [model.predict(_X) for _X in X_train_by_shot]

indices_train = [range(length-1,len(_y) + length - 1) for _y in y_train_by_shot]
indices_test = [range(length-1,len(_y)+length-1 ) for _y in y_test_by_shot]


savez('ttd_results',ttd=ttd_by_shot,ttd_prime = ttd_prime,ttd_prime_test = ttd_prime_test,
    ttd_prime_train = ttd_prime_train, indices_train = indices_train,indices_test = indices_test)

if plotting:
    print('plotting results')
    plot(ttd)
    plot(ttd_prime)
    plot(indices_test,ttd_prime_test,'g')
    plot(indices_train,ttd_prime_train,'r')
    savefig('plot.png')
    #plot(y_train,'.')
    #show()


