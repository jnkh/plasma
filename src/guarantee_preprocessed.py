'''
#########################################################
Assuming all shot files have been downloaded, this script
guarantees all preprocessing and produces ready-to-use shot
lists and loader objects.

The user should then use

nn = Normalizer(conf)
nn.train()
loader = Loader(conf,nn)
shot_list_train,shot_list_validate,shot_list_test = guarantee_preprocessed.load_shotlists(conf)

to load these into an application


Dependencies:
conf.py: configuration of model,training,paths, and data

Author: Julian Kates-Harbeck, jkatesharbeck@g.harvard.edu

This work was supported by the DOE CSGF program.
#########################################################
'''
#system
from __future__ import print_function
import math,os,sys,time,datetime,os.path
import dill
import numpy as np





def get_shot_list_path(conf):
    return conf['paths']['base_path'] + 'data/normalization/shot_lists.npz'


def save_shotlists(conf,shot_list_train,shot_list_validate,shot_list_test):
    path = get_shot_list_path(conf)
    np.savez(path,shot_list_train=shot_list_train,shot_list_validate=shot_list_validate,shot_list_test=shot_list_test)

def load_shotlists(conf):
    path = get_shot_list_path(conf)
    data = np.load(path)
    shot_list_train = data['shot_list_train'][()]
    shot_list_validate = data['shot_list_validate'][()]
    shot_list_test = data['shot_list_test'][()]
    return shot_list_train,shot_list_validate,shot_list_test


def main():

    from conf import conf
    from pprint import pprint
    pprint(conf)
    from data_processing import Shot, ShotList, Normalizer, Preprocessor, Loader

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
    validation_frac = conf['training']['validation_frac']
    if validation_frac <= 0.0:
        print('Setting validation to a minimum of 0.05')
        validation_frac = 0.05
    shot_list_train,shot_list_validate = shot_list_train.split_direct(1.0-validation_frac,shuffle=True)
    print('validate: {} shots, {} disruptive'.format(len(shot_list_validate),shot_list_validate.num_disruptive()))
    print('training: {} shots, {} disruptive'.format(len(shot_list_train),shot_list_train.num_disruptive()))
    print('testing: {} shots, {} disruptive'.format(len(shot_list_test),shot_list_test.num_disruptive()))
    print("...done")

    save_shotlists(conf,shot_list_train,shot_list_validate,shot_list_test)
    #####################################################
    ####################Normalization####################
    #####################################################


    print("normalization",end='')
    nn = Normalizer(conf)
    nn.train()
    loader = Loader(conf,nn)
    print("...done")



if __name__ == "__main__":
    main() 

