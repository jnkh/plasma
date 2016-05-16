from numpy import log10
#paths#
base_path = '/p/datad/jkatesha/'#'/p/datad/jkatesha/' #base_path = '../'
signals_dirs = ['jpf/da/c2-ipla','jpf/da/c2-loca','jpf/db/b5r-ptot>out',
                        'jpf/df/g1r-lid:003','jpf/gs/bl-li<s','jpf/gs/bl-fdwdt<s',
                        'jpf/gs/bl-ptot<s','jpf/gs/bl-wmhd<s']


def remap_target(ttd,T_warning,as_array_of_shots=True):
    binary_ttd = 0*ttd
    mask = ttd < log10(T_warning)
    binary_ttd[mask] = 1.0
    binary_ttd[~mask] = 0.0
    return binary_ttd


conf = {
    'paths': {
        'base_path' : base_path,
        'signal_prepath' : base_path + 'data/signal_data/jet/',
        'signals_dirs' : signals_dirs,
        'shot_files' : ['CWall_clear.txt','CFC_unint.txt'],#['mixed_list.txt',long_list_C.txt','short_list.txt','BeWall_clear.txt']
        'shot_files_test' : ['BeWall_clear.txt','ILW_unint.txt'],
        'shot_list_dir' : base_path + 'data/shot_lists/',
        #processed data
        'processed_prepath' : base_path + 'data/processed_shots/',
        'results_prepath' : base_path + 'data/results/',
   },

   'data': {
        'recompute' : False,
        #'recompute_minmax' : False
        'num_signals' : len(signals_dirs),
        'current_index' : 0,
        'plotting' : False,
        #train/validate split
        #how many shots to use
        'use_shots' : 2000,
        #normalization timescale
        'dt' : 0.001,
        #maximum TTD considered
        'T_max' : 2,
        'T_warning' : 0.4,
        'current_thresh' : 750000,
        'ttd_remapper' : remap_target,
   },

   'model': {
        #length of LSTM memory
        'length' : 100,
        'skip' : 1,
        #hidden layer size
        'rnn_size' : 20,
        'rnn_type' : 'LSTM',
        'optimizer' : 'adam',
        'dropout_prob' : 0.1,
    },

    'training': {
        'as_array_of_shots':True,
        'shuffle_training' : True,
        'train_frac' : 0.004,
        'batch_size_large' : 2048,
        'batch_size_small' : 2048,
        'batch_size' : 256,
        'num_shots_at_once' :  30,
        'num_epochs' : 1,
        'evaluate' : False,
   },
}
