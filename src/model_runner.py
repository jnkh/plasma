from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import pylab as pl


import numpy as np
from conf import conf
from itertools import imap


import time,sys,os
from functools import partial
import pathos.multiprocessing as mp

from data_processing import Loader
from performance_analysis_utils import PerformanceAnalyzer




def train(conf,shot_list_train,loader):

    np.random.seed(1)

    validation_losses = []
    validation_roc = []
    training_losses = []
    if conf['training']['validation_frac'] > 0.0:
        shot_list_train,shot_list_validate = shot_list_train.split_direct(1.0-conf['training']['validation_frac'],shuffle=True)
        print('validate: {} shots, {} disruptive'.format(len(shot_list_validate),shot_list_validate.num_disruptive()))
    print('training: {} shots, {} disruptive'.format(len(shot_list_train),shot_list_train.num_disruptive()))
    ##Need to import later because accessing the GPU from several processes via multiprocessing
    ## gives weird errors.
    os.environ['THEANO_FLAGS'] = 'device=gpu,floatX=float32'#,mode=NanGuardMode'
    import theano
    from keras.utils.generic_utils import Progbar 
    from keras import backend as K
    import model_builder #from model_builder import ModelBuilder, LossHistory

    print('Build model...',end='')
    builder = model_builder.ModelBuilder(conf)
    train_model,test_model = builder.build_train_test_models()
    print('...done')

    #load the latest epoch we did. Returns -1 if none exist yet
    e = builder.load_model_weights(train_model)

    num_epochs = conf['training']['num_epochs']
    num_at_once = conf['training']['num_shots_at_once']
    lr_decay = conf['model']['lr_decay']
    lr = conf['model']['lr']
    print('{} epochs left to go'.format(num_epochs - 1 - e))
    while e < num_epochs-1:
        e += 1
        print('\nEpoch {}/{}'.format(e+1,num_epochs))
        pbar =  Progbar(len(shot_list_train))

        #shuffle during every iteration
        shot_list_train.shuffle() 
        shot_sublists = shot_list_train.sublists(num_at_once)
        training_losses_tmp = []

        #decay learning rate each epoch:
        K.set_value(train_model.optimizer.lr, lr*lr_decay**(e))
        print('Learning rate: {}'.format(train_model.optimizer.lr.get_value()))
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

                pbar.add(1.0*len(shot_sublist)/len(X_list), values=[("train loss", train_loss)])
                loader.verbose=False#True during the first iteration
        sys.stdout.flush()
        training_losses.append(np.mean(training_losses_tmp))
        builder.save_model_weights(train_model,e)

        if conf['training']['validation_frac'] > 0.0:
            roc_area,loss = make_predictions_and_evaluate_gpu(conf,shot_list_validate,loader)
            validation_losses.append(loss)
            validation_roc.append(roc_area)

        print('=========Summary========')
        print('Training Loss: {:.3e}'.format(training_losses[-1]))
        if conf['training']['validation_frac'] > 0.0:
            print('Validation Loss: {:.3e}'.format(validation_losses[-1]))
            print('Validation ROC: {:.4f}'.format(validation_roc[-1]))


    # plot_losses(conf,[training_losses],builder,name='training')
    if conf['training']['validation_frac'] > 0.0:
        plot_losses(conf,[training_losses,validation_losses,validation_roc],builder,name='training_validation_roc')
    print('...done')




def plot_losses(conf,losses_list,builder,name=''):
    unique_id = builder.get_unique_id()
    savedir = 'losses'
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    save_path = os.path.join(savedir,'{}_loss_{}.png'.format(name,unique_id))
    pl.figure()
    for losses in losses_list:
        pl.semilogy(losses)
    pl.xlabel('Epoch')
    pl.ylabel('Loss')
    pl.grid()
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

    os.environ['THEANO_FLAGS'] = 'device=gpu,floatX=float32' #=cpu
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
    shot_sublists = shot_list.sublists(conf['model']['pred_batch_size'],shuffle=False,equal_size=True)
    for (i,shot_sublist) in enumerate(shot_sublists):
        X,y,shot_lengths,disr = loader.load_as_X_y_pred(shot_sublist)
        #load data and fit on data
        y_p = model.predict(X,
            batch_size=conf['model']['pred_batch_size'])
        model.reset_states()
        y_p = loader.batch_output_to_array(y_p)
        y = loader.batch_output_to_array(y)
        #cut arrays back
        y_p = [arr[:shot_lengths[j]] for (j,arr) in enumerate(y_p)]
        y = [arr[:shot_lengths[j]] for (j,arr) in enumerate(y)]

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



def make_predictions_and_evaluate_gpu(conf,shot_list,loader):
    y_prime,y_gold,disruptive = make_predictions_gpu(conf,shot_list,loader)
    analyzer = PerformanceAnalyzer(conf=conf)
    roc_area = analyzer.get_roc_area(y_prime,y_gold,disruptive)
    loss = get_loss_from_list(y_prime,y_gold,conf['data']['target'].loss)
    return roc_area,loss

def make_evaluations_gpu(conf,shot_list,loader):
    os.environ['THEANO_FLAGS'] = 'device=gpu,floatX=float32' #=cpu
    import theano
    from keras.utils.generic_utils import Progbar 
    from model_builder import ModelBuilder
    builder = ModelBuilder(conf) 

    y_prime = []
    y_gold = []
    disruptive = []
    batch_size = min(len(shot_list),conf['model']['pred_batch_size'])

    pbar =  Progbar(len(shot_list))
    print('evaluating {} shots using batchsize {}'.format(len(shot_list),batch_size))

    shot_sublists = shot_list.sublists(batch_size,equal_size=False)
    all_metrics = []
    all_weights = []
    for (i,shot_sublist) in enumerate(shot_sublists):
        batch_size = len(shot_sublist)
        model = builder.build_model(True,custom_batch_size=batch_size)
        builder.load_model_weights(model)
        model.reset_states()
        X,y,shot_lengths,disr = loader.load_as_X_y_pred(shot_sublist,custom_batch_size=batch_size)
        #load data and fit on data
        all_metrics.append(model.evaluate(X,y,batch_size=batch_size,verbose=False))
        all_weights.append(batch_size)
        model.reset_states()

        pbar.add(1.0*len(shot_sublist))
        loader.verbose=False#True during the first iteration

    if len(all_metrics) > 1:
        print('evaluations all: {}'.format(all_metrics))
    loss = np.average(all_metrics,weights = all_weights)
    print('Evaluation Loss: {}'.format(loss))
    return loss 



def get_loss_from_list(y_pred_list,y_gold_list,mode):
    return np.mean([get_loss(yp,yg,mode) for yp,yg in zip(y_pred_list,y_gold_list)])

def get_loss(y_pred,y_gold,mode):
    if mode == 'mae':
        return np.mean(np.abs(y_pred-y_gold))
    elif mode == 'binary_crossentropy':
        return np.mean(- (y_gold*np.log(y_pred) + (1-y_gold)*np.log(1 - y_pred)))
    elif mode == 'mse':
        return np.mean((y_pred-y_gold)**2)
    elif mode == 'hinge':
        return np.mean(np.maximum(0.0,1  - y_pred*y_gold))
    elif mode == 'squared_hinge':
        return np.mean(np.maximum(0.0,1  - y_pred*y_gold)**2)
    else:
        print('mode not recognized')
        exit(1)







