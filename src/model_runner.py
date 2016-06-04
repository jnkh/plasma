

from __future__ import print_function
import numpy as np
from data_processing import Loader
import os
from conf import conf
from itertools import imap


def train(conf,shot_list_train,loader):

    np.random.seed(5)
    ##Need to import later because accessing the GPU from several processes via multiprocessing
    ## gives weird errors.
    os.environ["THEANO_FLAGS"] = "device=gpu1"
    import theano
    from keras.utils.generic_utils import Progbar 
    import model_builder #from model_builder import ModelBuilder, LossHistory

    print('Build model...',end='')
    builder = model_builder.ModelBuilder(conf)
    train_model,test_model = builder.build_train_test_models()
    print('...done')

    #load the latest epoch we did. Returns -1 if none exist yet
    e = builder.load_model_weights(train_model)

    if conf['training']['data_parallel']:
        #elephas
        from pyspark import SparkConf, SparkContext
        sc = SparkContext()
        from elephas.utils.rdd_utils import to_simple_rdd
        from elephas.spark_model import SparkModel
        from elephas import optimizers as elephas_optimizers
        adam = elephas_optimizers.Adam()
        spark_model = SparkModel(sc,train_model,optimizer=adam,frequency='batch',
            mode='synchronous',num_workers=2)


    num_epochs = conf['training']['num_epochs']
    num_at_once = conf['training']['num_shots_at_once']
    print('{} epochs left to go'.format(num_epochs - 1 - e))
    while e < num_epochs-1:
        e += 1
        print('Epoch {}/{}'.format(e+1,num_epochs))
        pbar =  Progbar(len(shot_list_train))

        #shuffle during every iteration
        shot_list_train.shuffle() 
        shot_sublists = shot_list_train.sublists(num_at_once)
        for (i,shot_sublist) in enumerate(shot_sublists):
            X_list,y_list = loader.load_as_X_y_list(shot_sublist)
            for j,(X,y) in enumerate(zip(X_list,y_list)):
                history = model_builder.LossHistory()

                if conf['training']['data_parallel']:
                    rdd = to_simple_rdd(sc,X,y)
                    spark_model.train(rdd,
                    batch_size=Loader.get_batch_size(conf['training']['batch_size'],prediction_mode=False),
                    nb_epoch=1,verbose=1,
                    validation_split=0.0)

                else:
                    #load data and fit on data
                    train_model.fit(X,y,
                        batch_size=Loader.get_batch_size(conf['training']['batch_size'],prediction_mode=False),
                        nb_epoch=1,shuffle=False,verbose=0,
                        validation_split=0.0,callbacks=[history])
                    train_model.reset_states()

                # print('Shots {}/{}'.format(i*num_at_once + j*1.0*len(shot_sublist)/len(X_list),len(shot_list_train)))
                pbar.add(1.0*len(shot_sublist)/len(X_list), values=[("train loss", np.mean(history.losses))])
                loader.verbose=False#True during the first iteration

        if conf['training']['data_parallel']:
            builder.save_model_weights(spark_model,e)
        else:
            builder.save_model_weights(train_model,e)

        #validation
        if conf['training']['evaluate']:
            builder.load_model_weights(test_model)
            for (i,shot) in enumerate(shot_list_train):
                X,y = loader.load_as_X_y(shot,prediction_mode=True)
                print(test_model.evaluate(X,y,batch_size=Loader.get_batch_size(conf['training']['batch_size'],prediction_mode=True)))

    print('...done')



import model_builder
import time,sys
from functools import partial
import pathos.multiprocessing as mp


def make_predictions(conf,shot_list,builder,loader):


    os.environ["THEANO_FLAGS"] = "device=gpu" #=cpu
    import theano
    from keras.utils.generic_utils import Progbar 

    y_prime = []
    y_gold = []
    disruptive = []

    _,model = builder.build_train_test_models()
    builder.load_model_weights(model)

    start_time = time.time()
    fn = partial(make_single_prediction,builder=builder,loader=loader,model=model)


    for (i,(y_p,y,is_disruptive)) in enumerate(imap(fn,shot_list)):
        print('Shot {}/{}'.format(i,len(shot_list)))
        y_prime.append(y_p)
        y_gold.append(y)
        disruptive.append(is_disruptive)
    print('Finished Predictions in {} seconds'.format(time.time()-start_time))
    return y_prime,y_gold,disruptive




def make_single_prediction(shot,builder,loader,model):
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
