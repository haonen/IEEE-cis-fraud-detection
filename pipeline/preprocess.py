"""
Preprocess data
"""
import logging
import sys
from .imputer import *
from .create_dummies import *
from .scaler import *


logger = logging.getLogger('start to transform the data')
ch = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.INFO)


def transform(config, X_train, X_test, low_dt, high_dt):
    '''
    Impute, hot encoding and scaling
    Inputs:
        config: the yml file to get parameters
        X_train: train df
        X_test: test df
    Returns:
        X_train and an imputer
    '''
    cat_cols = config['cols']
    k = config['dummy']['k'][0]
    
    logger.info('begin to transform')
    logger.info('begin to impute')
    #initializing impute_table
    imputer = Imputer(config['impute_table_name'])
    imputer.feed_impute_table(X_train, low_dt, high_dt, cat_cols)
        
    #imputing  
    X_train = imputer.impute_cat_single_table(X_train, cat_cols)
    X_train = imputer.impute_cond_single_table(X_train, cat_cols)
       
    #hot encoding
    logger.info('begin to get dummies')
    X_train, X_test = get_dummies(X_train, X_test, cat_cols, k)
    
    #scaling
    logger.info('begin to scale')
    continuous_columns = list(set(X_train.columns) - set(cat_cols))
    #continuous_columns.remove('transactionid')
    #continuous_columns.remove('transactiondt')
    logger.info('start to scaling')
    X_train, X_test = min_max_transformation(X_train, X_test, continuous_columns)
    
    return X_train, X_test

def transform_train(config, X_train, low_dt, high_dt):
    '''
    Impute, hot encoding and scaling
    Inputs:
        config: the yml file to get parameters
        X_train: train df
    Returns:
        X_train and transform config
    '''
    cat_cols = config['cols']
    k = config['dummy']['k'][0]
    
    logger.info('begin to transform')
    logger.info('begin to impute')
    #initializing impute_table
    imputer = Imputer(config['impute_table_name'])
    imputer.feed_impute_table(X_train, low_dt, high_dt, cat_cols)
        
    #imputing  
    X_train = imputer.impute_cat_single_table(X_train, cat_cols)
    X_train = imputer.impute_cond_single_table(X_train, cat_cols)
       
    #hot encoding
    logger.info('begin to get dummies')
    X_train, dummyfillers = get_dummies_of_train(X_train, cat_cols, k)
    
    #scaling
    logger.info('begin to scale')
    continuous_columns = list(set(X_train.columns) - set(cat_cols))
    #continuous_columns.remove('transactionid')
    #continuous_columns.remove('transactiondt')
    logger.info('start to scaling')
    X_train, scaler = min_max_transformation_train(X_train, continuous_columns)
    
    transform_config = {'imputer': imputer, 'scaler': scaler, 'dummyfillers': dummyfillers}
    return X_train, transform_config

def transform_test(config, X_test, transform_config):
    '''
    Impute, hot encoding and scaling test data with transform_config
    Inputs:
        confit:
        X_test: test df
        transform_config: config of transform build by transform_train
    Returns:
        X_test
    '''
    cat_cols = config['cols']

    logger.info('begin to transform test data')
    # impute
    logger.info('begin to impute test data')
    imputer = transform_config['imputer']
    X_test = imputer.impute_cat_single_table(X_test, cat_cols)
    X_test = imputer.impute_cond_single_table(X_test, cat_cols)

    # hot encoding
    logger.info('begin to get dummies for test data')
    dummyfillers = transform_config['dummyfillers']
    X_test = get_dummies_of_test(X_test, cat_cols, dummyfillers)

    # scaler
    logger.info('begin to scale test data')
    continuous_columns = list(set(X_test.columns) - set(cat_cols))
    scaler = transform_config['scaler']
    X_test = min_max_transformation_test(X_test, scaler, continuous_columns)
    return X_test
