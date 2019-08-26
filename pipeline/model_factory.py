'''
Model factory for the pipeline
'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import sklearn.ensemble

import yaml
from collections import OrderedDict
from itertools import product
import logging
import sys
import numpy as np
import argparse
import os

logger = logging.getLogger('generating models')
ch = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.INFO)


def get_models(config):
    '''
    model factory generate the next aviable model 
    from the config file
    
    Input: 
        config: OrdedDict with the key as the name of the model, value as the parameters 
    Return:
        A iterable of models
    ''' 
    logger.info('begin to generate the models')
    #pdb.set_trace()
    for name, params in config.items():
        constructor = globals()[name]
        if name in dir(sklearn.ensemble):
            if 'base_estimator' in params:
                if type(params['base_estimator']) is not list:
                    base_estimator_config = params['base_estimator']
                    base_estimator = list(get_models(base_estimator_config))
                    assert len(base_estimator) == 1, "Too much estimators are generated for AdaBoost"
                    print("Base estimator for Adaboost")
                    print(base_estimator[0][1])
                    params['base_estimator'] = [base_estimator[0][1]]

        if name == 'GaussianNB':
            models =  [constructor()]
        else:
            models = [constructor(**dict(zip(params.keys(),vals))) for vals in product(*params.values())]
        for model in models:
            logger.info('{} is delivering out'.format(model))
            yield name, model
