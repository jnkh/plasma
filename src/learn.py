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

from __future__ import print_function
import datetime,time,os.path,sys
import dill
from functools import partial

#matplotlib
import matplotlib
matplotlib.use('Agg')
import numpy as np

#my classes
from conf import conf
from pprint import pprint
pprint(conf)
from data_processing import Shot, ShotList, Normalizer, Preprocessor, Loader

if conf['data']['normalizer'] == 'minmax':
    from data_processing import MinMaxNormalizer as Normalizer #performs !much better than minmaxnormalizer
elif conf['data']['normalizer'] == 'meanvar':
    from data_processing import MeanVarNormalizer as Normalizer #performs !much better than minmaxnormalizer
elif conf['data']['normalizer'] == 'var':
    from data_processing import VarNormalizer as Normalizer #performs !much better than minmaxnormalizer
else:
    print('unkown normalizer. exiting')
    exit(1)

num_epochs = conf['training']['num_epochs']
shot_list_dir = conf['paths']['shot_list_dir']
shot_files = conf['paths']['shot_files']
shot_files_test = conf['paths']['shot_files_test']
train_frac = conf['training']['train_frac']
stateful = conf['model']['stateful']
# if stateful: 
#     batch_size = conf['model']['length']
# else:
#     batch_size = conf['training']['batch_size_large']

np.random.seed(1)
#####################################################
####################PREPROCESSING####################
#####################################################

print("preprocessing all shots",end='')
pp = Preprocessor(conf)
pp.clean_shot_lists()
shot_list = pp.preprocess_all()
sorted(shot_list)
shot_list_train,shot_list_test = shot_list.split_train_test(conf)
num_shots = len(shot_list_train) + len(shot_list_test)
print("...done")


#####################################################
####################Normalization####################
#####################################################


print("normalization",end='')
nn = Normalizer(conf)
nn.train()
loader = Loader(conf,nn)
print("...done")


print('Training on {} shots, testing on {} shots'.format(len(shot_list_train),len(shot_list_test)))



def train(conf,shot_list_train,loader):
    #####################################################
    ####################Training#########################
    #####################################################

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




import multiprocessing as old_mp
import pathos.multiprocessing as mp
p = old_mp.Process(target = train,args=(conf,shot_list_train,loader))
p.start()
p.join()




os.environ["THEANO_FLAGS"] = "device=cpu"
import theano
from keras.utils.generic_utils import Progbar 
import model_builder #from model_builder import ModelBuilder, LossHistory
builder = model_builder.ModelBuilder(conf)

#####################################################
####################Evaluating#######################
#####################################################

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



def make_predictions(conf,shot_list,builder,loader):
    y_prime = []
    y_gold = []
    disruptive = []

    _,model = builder.build_train_test_models()
    weights_path = builder.get_latest_save_path()

    pool = mp.Pool(1)
    print('running in parallel on {} processes'.format(pool._processes))
    start_time = time.time()
    #force compilation
    fn = partial(make_single_prediction,builder=builder,loader=loader,weights_path=weights_path)

    for (i,(y_p,y,is_disruptive)) in enumerate(pool.imap(fn,shot_list)):
        sys.stdout.write('Shot {}/{}'.format(i,len(shot_list)))
        y_prime.append(y_p)
        y_gold.append(y)
        disruptive.append(is_disruptive)
    pool.close()
    pool.join()
    print('Finished Predictions in {} seconds'.format(time.time()-start_time))
    return y_prime,y_gold,disruptive




def make_single_prediction(shot,builder,loader,weights_path):
    _,model = builder.build_train_test_models()
    model.load_weights(weights_path)
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


y_prime_train,y_gold_train,disruptive_train = make_predictions(conf,shot_list_train,builder,loader)
y_prime_test,y_gold_test,disruptive_test = make_predictions(conf,shot_list_test,builder,loader)



disruptive_train = np.array(disruptive_train)
disruptive_test = np.array(disruptive_test)

y_gold = y_gold_train + y_gold_test
y_prime = y_prime_train + y_prime_test
disruptive = np.concatenate((disruptive_train,disruptive_test))

shot_list.make_light()
shot_list_test.make_light()
shot_list_train.make_light()

save_str = 'results_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
np.savez(conf['paths']['results_prepath']+save_str,
    y_gold=y_gold,y_gold_train=y_gold_train,y_gold_test=y_gold_test,
    y_prime=y_prime,y_prime_train=y_prime_train,y_prime_test=y_prime_test,
    disruptive=disruptive,disruptive_train=disruptive_train,disruptive_test=disruptive_test,
    shot_list=shot_list,shot_list_train=shot_list_train,shot_list_test=shot_list_test,
    conf = conf)



print('finished.')





