'''
#########################################################
This file trains a deep learning model to predict
disruptions on time series data from plasma discharges.

Dependencies:
conf.py: configuration of model,training,paths, and data
model_builder.py: logic to construct the ML architecture
data_processing.py: classes to handle data processing

Author: Julian Kates-Harbeck, jkatesharbeck@g.harvard.edu

This work was supported by the DOE CSGF program.
#########################################################
'''
#system
from __future__ import print_function
import math,os,sys,time,datetime,os.path,random
import dill
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functools import partial
import itertools

from mpi_model import MPIModel,MPISGD,MPIAdam,print_unique,print_all

#import keras sequentially because it otherwise reads from ~/.keras/keras.json with too many threads.
#from mpi_launch_tensorflow import get_mpi_task_index 
from mpi4py import MPI
comm = MPI.COMM_WORLD
task_index = comm.Get_rank()
num_workers = comm.Get_size()
NUM_GPUS = 4
MY_GPU = task_index % NUM_GPUS
base_compile_dir = '/scratch/jk7/tmp/{}'.format(task_index)
os.environ['THEANO_FLAGS'] = 'device=gpu{},floatX=float32,base_compiledir={}'.format(MY_GPU,base_compile_dir)#,mode=NanGuardMode'
import theano
#import keras
for i in range(num_workers):
  comm.Barrier()
  if i == task_index:
    print('[{}] importing Keras'.format(task_index))
    from keras import backend as K
    from keras.layers import Input,Dense, Dropout
    from keras.layers.recurrent import LSTM
    from keras.layers.wrappers import TimeDistributed
    from keras.models import Model
    from keras.optimizers import SGD

#my classes
from conf import conf
from pprint import pprint
if task_index == 0:
    pprint(conf)
from data_processing import Shot, ShotList, Normalizer, Preprocessor, Loader
import model_builder
from model_runner import make_predictions_gpu,make_predictions_and_evaluate_gpu
import guarantee_preprocessed

if conf['data']['normalizer'] == 'minmax':
    from data_processing import MinMaxNormalizer as Normalizer
elif conf['data']['normalizer'] == 'meanvar':
    from data_processing import MeanVarNormalizer as Normalizer 
elif conf['data']['normalizer'] == 'var':
    from data_processing import VarNormalizer as Normalizer #performs !much better than minmaxnormalizer
elif conf['data']['normalizer'] == 'averagevar':
    from data_processing import AveragingVarNormalizer as Normalizer #performs !much better than minmaxnormalizer
else:
    print('unkown normalizer. exiting')
    exit(1)

shot_list_dir = conf['paths']['shot_list_dir']
shot_files = conf['paths']['shot_files']
shot_files_test = conf['paths']['shot_files_test']
train_frac = conf['training']['train_frac']
stateful = conf['model']['stateful']
# if stateful: 
#     batch_size = conf['model']['length']
# else:
#     batch_size = conf['training']['batch_size_large']

np.random.seed(task_index)
random.seed(task_index)


print("normalization",end='')
nn = Normalizer(conf)
nn.train()
loader = Loader(conf,nn)
print("...done")


shot_list_train,shot_list_validate,shot_list_test = guarantee_preprocessed.load_shotlists(conf)



def concatenate_sublists(superlist):
    return list(itertools.chain.from_iterable(superlist))


def mpi_make_predictions(conf,shot_list,loader):

    builder = model_builder.ModelBuilder(conf) 

    y_prime = []
    y_gold = []
    disruptive = []

    _,model = builder.build_train_test_models()
    builder.load_model_weights(model)
    model.reset_states()

    pbar =  Progbar(len(shot_list))
    shot_sublists = shot_list.sublists(conf['model']['pred_batch_size'],equal_size=True)

    y_prime_global = []
    y_gold_global = []
    disruptive_global = []



    for (i,shot_sublist) in enumerate(shot_sublists):

        if i % num_workers == task_index:
            X,y,shot_lengths,disr = loader.load_as_X_y_pred(shot_sublist)
            #load data and fit on data
            y_p = model.predict(X,
                batch_size=conf['model']['pred_batch_size'])
            model.reset_states()
            y_p = loader.batch_output_to_array(y_p)
            y = loader.batch_output_to_array(y)
            #cut arrays back
            y_p = [arr[:shot_lengths[i]] for (i,arr) in enumerate(y_p)]
            y = [arr[:shot_lengths[i]] for (i,arr) in enumerate(y)]

            # print('Shots {}/{}'.format(i*num_at_once + j*1.0*len(shot_sublist)/len(X_list),len(shot_list_train)))
            pbar.add(1.0*len(shot_sublist))
            loader.verbose=False#True during the first iteration
            y_prime += y_p
            y_gold += y
            disruptive += disr

        if i % num_workers == num_workers -1:
            comm.Barrier()
            y_prime_global += concatenate_sublists(comm.allgather(y_prime))
            y_gold_global += concatenate_sublists(comm.allgather(y_gold))
            disruptive_global += concatenate_sublists(comm.allgather(disruptive))


    y_prime_global = y_prime_global[:len(shot_list)]
    y_gold_global = y_gold_global[:len(shot_list)]
    disruptive_global = disruptive_global[:len(shot_list)]
    return y_prime_global,y_gold_global,disruptive_global


def mpi_make_predictions_and_evaluate(conf,shot_list,loader):
    y_prime,y_gold,disruptive = mpi_make_predictions(conf,shot_list,loader)
    analyzer = PerformanceAnalyzer(conf=conf)
    roc_area = analyzer.get_roc_area(y_prime,y_gold,disruptive)
    loss = get_loss_from_list(y_prime,y_gold,conf['data']['target'].loss)
    return roc_area,loss




def mpi_train(conf,shot_list_train,shot_list_validate,loader):
    validation_losses = []
    validation_roc = []
    # training_losses = []

    builder = model_builder.ModelBuilder(conf)
    train_model,test_model = builder.build_train_test_models()

    #load the latest epoch we did. Returns -1 if none exist yet
    e = builder.load_model_weights(train_model)

    num_epochs = conf['training']['num_epochs']
    lr_decay = conf['model']['lr_decay']
    batch_size = conf['training']['batch_size']
    lr = conf['model']['lr']
    optimizer = MPIAdam(lr=lr)
    print('{} epochs left to go'.format(num_epochs - 1 - e))
    batch_generator = partial(loader.training_batch_generator,shot_list=shot_list_train,loader=loader)

    mpi_model = MPIModel(train_model,optimizer,comm,batch_generator,batch_size,lr=lr,warmup_steps = 50)
    mpi_model.compile(loss=conf['data']['target'].loss)


    while e < num_epochs-1:
        e += 1
        mpi_model.set_lr(lr*lr_decay**e)
        print_unique('\nEpoch {}/{}'.format(e,num_epochs))

        mpi_model.train_epoch()

        loader.verbose=False#True during the first iteration
        if task_index == 0:
            builder.save_model_weights(train_model,e)


        roc_area,loss = mpi_make_predictions_and_evaluate(conf,shot_list_validate,loader)

        print_all('=========Summary========')
        # print('Training Loss: {:.3e}'.format(training_losses[-1]))
        print_all('Validation Loss: {:.3e}'.format(validation_losses[-1]))
        print_all('Validation ROC: {:.4f}'.format(validation_roc[-1]))

        print('...done')


        if task_index == 0:

            roc_area,loss = make_predictions_and_evaluate_gpu(conf,shot_list_validate,loader)
            validation_losses.append(loss)
            validation_roc.append(roc_area)

            print('=========Summary========')
            # print('Training Loss: {:.3e}'.format(training_losses[-1]))
            print('Validation Loss: {:.3e}'.format(validation_losses[-1]))
            print('Validation ROC: {:.4f}'.format(validation_roc[-1]))

            # plot_losses(conf,[training_losses],builder,name='training')
            # plot_losses(conf,[validation_losses,validation_roc],builder,name='training_validation_roc')
            print('...done')




mpi_train(conf,shot_list_train,shot_list_validate,loader)



from model_runner import make_predictions,make_predictions_gpu

#load last model for testing
print('saving results')
y_prime = []
y_prime_test = []
y_prime_train = []

y_gold = []
y_gold_test = []
y_gold_train = []

disruptive= []
disruptive_train= []
disruptive_test= []

# y_prime_train,y_gold_train,disruptive_train = make_predictions(conf,shot_list_train,loader)
# y_prime_test,y_gold_test,disruptive_test = make_predictions(conf,shot_list_test,loader)

y_prime_train,y_gold_train,disruptive_train = mpi_make_predictions(conf,shot_list_train,loader)
y_prime_test,y_gold_test,disruptive_test = mpi_make_predictions(conf,shot_list_test,loader)


if task_index == 0:
    disruptive_train = np.array(disruptive_train)
    disruptive_test = np.array(disruptive_test)

    y_gold = y_gold_train + y_gold_test
    y_prime = y_prime_train + y_prime_test
    disruptive = np.concatenate((disruptive_train,disruptive_test))

    shot_list_test.make_light()
    shot_list_train.make_light()

    save_str = 'results_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    np.savez(conf['paths']['results_prepath']+save_str,
        y_gold=y_gold,y_gold_train=y_gold_train,y_gold_test=y_gold_test,
        y_prime=y_prime,y_prime_train=y_prime_train,y_prime_test=y_prime_test,
        disruptive=disruptive,disruptive_train=disruptive_train,disruptive_test=disruptive_test,
        shot_list_train=shot_list_train,shot_list_test=shot_list_test,
        conf = conf)



    print('finished.')





