from numpy import log10
from keras.optimizers import SGD
#paths#
base_path = '/p/datad/jkatesha/'#'/p/datad/jkatesha/' #base_path = '../'
signals_dirs = ['jpf/da/c2-ipla', # Plasma Current [A]
                'jpf/da/c2-loca', # Mode Lock Amplitude [A]
                'jpf/db/b5r-ptot>out', #Radiated Power [W]
                'jpf/df/g1r-lid:003', #Density [m^-2]
                'jpf/gs/bl-li<s', #Plasma Internal Inductance
                'jpf/gs/bl-fdwdt<s', #Stored Diamagnetic Energy (time derivative) [W]
                'jpf/gs/bl-ptot<s', #total input power [W]
                'jpf/gs/bl-wmhd<s'] #unkown


def remap_target(ttd,T_warning,as_array_of_shots=True):
    binary_ttd = 0*ttd
    mask = ttd < log10(T_warning)
    binary_ttd[mask] = 1.0
    binary_ttd[~mask] = 0.0
    return binary_ttd


conf = {
    'paths': {
        'base_path' : base_path,
        #'signal_prepath' : base_path + 'data/signal_data/jet/',
        'signal_prepath' : base_path + 'data/signal_data/jet/',
        'signals_dirs' : signals_dirs,
        'shot_files' : ['CWall_clear.txt','CFC_unint.txt'],#['mixed_list1.txt',long_list_C.txt','short_list.txt','BeWall_clear.txt']
        'shot_files_test' : ['BeWall_clear.txt','ILW_unint.txt'],
        'shot_list_dir' : base_path + 'data/shot_lists/',
        #processed data
        'processed_prepath' : base_path + 'data/processed_shots/',
        'normalizer_path' : base_path + 'data/normalization/normalization.npz',
        'results_prepath' : base_path + 'data/results/',
        'model_save_path' : './tmp/'
   },

   'data': {
        'recompute' : False,
        'recompute_normalization' : False,
        #'recompute_minmax' : False
        'num_signals' : len(signals_dirs),
        'current_index' : 0,
        'plotting' : False,
        #train/validate split
        #how many shots to use
        'use_shots' : 1000,
        #normalization timescale
        'dt' : 0.001,
        #maximum TTD considered
        'T_max' : 2,
        'T_warning' : 1.0,
        'current_thresh' : 750000,
        'ttd_remapper' : remap_target,
   },

   'model': {
        #length of LSTM memory
        'pred_length' : 400,
        'length' : 100,
        'skip' : 1,
        #hidden layer size
        'rnn_size' : 100,
        'rnn_type' : 'LSTM',
        'optimizer' : 'adam',
        'loss' : 'binary_crossentropy',
        'stateful' : True,
        'return_sequences' : True,
        'dropout_prob' : 0.1,
    },

    'training': {
        'as_array_of_shots':True,
        'shuffle_training' : True,
        'train_frac' : 0.75,
        'batch_size' : 256, #100
        'max_patch_length' : 10000, #THIS WAS THE CULPRIT FOR NO TRAINING!
        'num_shots_at_once' :  25,
        'num_epochs' : 10,
        'evaluate' : False,
        'use_mock_data' : False,
        'data_parallel' : False,
   },
}
