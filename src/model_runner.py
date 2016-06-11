from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import pylab as pl


import numpy as np
from data_processing import Loader
import os
from conf import conf
from itertools import imap


import time,sys
from functools import partial
import pathos.multiprocessing as mp


def train(conf,shot_list_train,loader):

    np.random.seed(1)

    validation_losses = []
    training_losses = []
    if conf['training']['validation_frac'] > 0.0:
        shot_list_train,shot_list_validate = shot_list_train.split_direct(1.0-conf['training']['validation_frac'],shuffle=True)
        print('validate: {} shots, {} disruptive'.format(len(shot_list_validate),shot_list_validate.num_disruptive()))
    print('training: {} shots, {} disruptive'.format(len(shot_list_train),shot_list_train.num_disruptive()))
    ##Need to import later because accessing the GPU from several processes via multiprocessing
    ## gives weird errors.
    os.environ['THEANO_FLAGS'] = 'device=gpu'
    import theano
    from keras.utils.generic_utils import Progbar 
    import model_builder #from model_builder import ModelBuilder, LossHistory

    print('Build model...',end='')
    builder = model_builder.ModelBuilder(conf)
    train_model,test_model = builder.build_train_test_models()
    print('...done')

    #load the latest epoch we did. Returns -1 if none exist yet
    e = builder.load_model_weights(train_model)


    num_epochs = conf['training']['num_epochs']
    num_at_once = conf['training']['num_shots_at_once']
    print('{} epochs left to go'.format(num_epochs - 1 - e))
    while e < num_epochs-1:
        e += 1
        print('\nEpoch {}/{}'.format(e+1,num_epochs))
        pbar =  Progbar(len(shot_list_train))

        #shuffle during every iteration
        shot_list_train.shuffle() 
        shot_sublists = shot_list_train.sublists(num_at_once)
        training_losses_tmp = []
        for (i,shot_sublist) in enumerate(shot_sublists):
            X_list,y_list = loader.load_as_X_y_list(shot_sublist)
            for j,(X,y) in enumerate(zip(X_list,y_list)):
                history = model_builder.LossHistory()
                #load data and fit on data
                train_model.fit(X,y,
                    batch_size=Loader.get_batch_size(conf['training']['batch_size'],prediction_mode=False),
                    nb_epoch=1,shuffle=False,verbose=0,
                    validation_split=0.0,callbacks=[history])
                train_model.reset_states()
                train_loss = np.mean(history.losses)
                training_losses_tmp.append(train_loss)

                # print('Shots {}/{}'.format(i*num_at_once + j*1.0*len(shot_sublist)/len(X_list),len(shot_list_train)))
                pbar.add(1.0*len(shot_sublist)/len(X_list), values=[("train loss", train_loss)])
                loader.verbose=False#True during the first iteration

        training_losses.append(np.mean(training_losses_tmp))
        builder.save_model_weights(train_model,e)

        if conf['training']['validation_frac'] > 0.0:
            validation_losses.append(make_evaluations_gpu(conf,shot_list_validate,loader))

    plot_losses(conf,training_losses,builder,name='training')
    if conf['training']['validation_frac'] > 0.0:
        plot_losses(conf,validation_losses,builder,name='validation')
    print('...done')




def plot_losses(conf,losses,builder,name=''):
    unique_id = builder.get_unique_id()
    savedir = 'losses'
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    save_path = os.path.join(savedir,'{}_loss_{}.png'.format(name,unique_id))
    pl.semilogy(losses)
    xlabel('Epoch')
    ylabel('Loss')
    pl.savefig(save_path)





def make_predictions(conf,shot_list,loader):

    os.environ['THEANO_FLAGS'] = 'device=cpu' #=cpu
    import theano
    from keras.utils.generic_utils import Progbar 
    from model_builder import ModelBuilder
    builder = ModelBuilder(conf) 
    


    y_prime = []
    y_gold = []
    disruptive = []

    _,model = builder.build_train_test_models()
    builder.load_model_weights(model)
    model_save_path = builder.get_latest_save_path()

    start_time = time.time()
    pool = mp.Pool()
    fn = partial(make_single_prediction,builder=builder,loader=loader,model_save_path=model_save_path)

    print('running in parallel on {} processes'.format(pool._processes))
    for (i,(y_p,y,is_disruptive)) in enumerate(pool.imap(fn,shot_list)):
    # for (i,(y_p,y,is_disruptive)) in enumerate(imap(fn,shot_list)):
        print('Shot {}/{}'.format(i,len(shot_list)))
        sys.stdout.flush()
        y_prime.append(y_p)
        y_gold.append(y)
        disruptive.append(is_disruptive)
    pool.close()
    pool.join()
    print('Finished Predictions in {} seconds'.format(time.time()-start_time))
    return y_prime,y_gold,disruptive




def make_single_prediction(shot,builder,loader,model_save_path):
    _,model = builder.build_train_test_models()
    model.load_weights(model_save_path)
    model.reset_states()
    X,y = loader.load_as_X_y(shot,prediction_mode=True)
    assert(X.shape[0] == y.shape[0])
    y_p = model.predict(X,batch_size=Loader.get_batch_size(conf['training']['batch_size'],prediction_mode=True),verbose=0)
    answer_dims = y_p.shape[-1]
    if conf['model']['return_sequences']:
        shot_length = y_p.shape[0]*y_p.shape[1]
    else:
        shot_length = y_p.shape[0]
    y_p = np.reshape(y_p,(shot_length,answer_dims))
    y = np.reshape(y,(shot_length,answer_dims))
    is_disruptive = shot.is_disruptive_shot()
    model.reset_states()
    return y_p,y,is_disruptive


def make_predictions_gpu(conf,shot_list,loader):

    os.environ['THEANO_FLAGS'] = 'device=gpu' #=cpu
    import theano
    from keras.utils.generic_utils import Progbar 
    from model_builder import ModelBuilder
    builder = ModelBuilder(conf) 

    y_prime = []
    y_gold = []
    disruptive = []

    _,model = builder.build_train_test_models()
    builder.load_model_weights(model)
    model.reset_states()

    pbar =  Progbar(len(shot_list))
    shot_sublists = shot_list.sublists(conf['model']['pred_batch_size'],equal_size=True)
    for (i,shot_sublist) in enumerate(shot_sublists):
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
    y_prime = y_prime[:len(shot_list)]
    y_gold = y_gold[:len(shot_list)]
    disruptive = disruptive[:len(shot_list)]
    return y_prime,y_gold,disruptive

def make_evaluations_gpu(conf,shot_list,loader):
    print('evaluating:')

    os.environ['THEANO_FLAGS'] = 'device=gpu' #=cpu
    import theano
    from keras.utils.generic_utils import Progbar 
    from model_builder import ModelBuilder
    builder = ModelBuilder(conf) 

    y_prime = []
    y_gold = []
    disruptive = []
    batch_size = min(len(shot_list),conf['model']['pred_batch_size'])

    print('evaluating {} shots using batchsize {}'.format(len(shot_list),batch_size))
    model = builder.build_model(True,custom_batch_size=batch_size)
    builder.load_model_weights(model)
    model.reset_states()

    pbar =  Progbar(len(shot_list))
    shot_sublists = shot_list.sublists(batch_size,equal_size=True)
    all_metrics = []
    for (i,shot_sublist) in enumerate(shot_sublists):
        X,y,shot_lengths,disr = loader.load_as_X_y_pred(shot_sublist,custom_batch_size=batch_size)
        #load data and fit on data
        all_metrics.append(model.evaluate(X,y,batch_size=batch_size))
        model.reset_states()

        pbar.add(1.0*len(shot_sublist))
        loader.verbose=False#True during the first iteration

    if len(all_metrics) > 1:
        print('evaluations all: {}'.format(all_metrics))
    print('evaluations mean: {}'.format(np.mean(all_metrics)))
    return np.mean(all_metrics)

