import targets as t
#paths#
base_path = '/tigress/jk7/'#'/p/datad/jkatesha/'#'/p/datad/jkatesha/' #base_path = '../'
signals_dirs = ['jpf/da/c2-ipla', # Plasma Current [A]
                'jpf/da/c2-loca', # Mode Lock Amplitude [A]
                'jpf/db/b5r-ptot>out', #Radiated Power [W]
                'jpf/df/g1r-lid:003', #Density [m^-2]
                'jpf/gs/bl-li<s', #Plasma Internal Inductance
                'jpf/gs/bl-fdwdt<s', #Stored Diamagnetic Energy (time derivative) [W]
                'jpf/gs/bl-ptot<s', #total input power [W]
                'jpf/gs/bl-wmhd<s'] #unkown

target = t.HingeTarget

conf = {
    'paths': {
        'base_path' : base_path,
        #'signal_prepath' : base_path + 'data/signal_data/jet/',
        'signal_prepath' : base_path + 'data/signal_data/jet/',
        'signals_dirs' : signals_dirs,
        'shot_files' : ['CWall_clear.txt','CFC_unint.txt'],#['mixed_list1.txt'],#['short_list.txt'],#['CWall_clear.txt','CFC_unint.txt'],#['mixed_list1.txt',long_list_C.txt','short_list.txt','BeWall_clear.txt']
        'shot_files_test' : ['BeWall_clear.txt','ILW_unint.txt'] ,#[],#['BeWall_clear.txt','ILW_unint.txt'],
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
        'use_shots' : 2000,
        #normalization timescale
        'dt' : 0.001,
        #maximum TTD considered
        'T_max' : 1000.0,
        'T_warning' : 1.0, #The shortest works best so far: less overfitting. log TTd prediction also works well. 0.5 better than 0.2
        'current_thresh' : 750000,
        'window_decay' : 10, #the characteristic decay length of the decaying moving average window
        'window_size' : 70, #the width of the actual window
        'target' : target,
        'normalizer' : 'averagevar',           #TODO optimize
   },

   'model': {
        #length of LSTM memory
        'pred_length' : 200,
        'pred_batch_size' : 128,
        'length' : 128,                     #TODO optimize
        'skip' : 1,
        #hidden layer size
        'rnn_size' : 100,                   #TODO optimize
        #size 100 slight overfitting, size 20 no overfitting. 200 is not better than 100. Prediction much better with size 100, size 20 cannot capture the data.
        'rnn_type' : 'LSTM',
        'rnn_layers' : 3,                   #TODO optimize
        # 'output_activation' : target.activation,
        'optimizer' : 'adam', #have not found a difference yet
        'clipnorm' : 10.0,
        'regularization' : 0.01,
        # 'loss' : target.loss, #binary crossentropy performs slightly better?
        'lr' : 0.00001,#None,#001, #lower better, at most 0.0001. 0.00001 is too low
        'lr_decay' : 0.5,
        'stateful' : True,
        'return_sequences' : True,
        'dropout_prob' : 0.3,
    },

    'training': {
        'as_array_of_shots':True,
        'shuffle_training' : True,
        'train_frac' : 0.5,
        'validation_frac' : 0.05,
        'batch_size' : 256, #100
        'max_patch_length' : 100000, #THIS WAS THE CULPRIT FOR NO TRAINING! Lower than 1000 performs very poorly
        'num_shots_at_once' :  200, #How many shots are we loading at once?
        'num_epochs' : 20,
        'use_mock_data' : False,
        'data_parallel' : False,
   },
}
