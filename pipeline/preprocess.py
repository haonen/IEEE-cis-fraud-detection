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
        X_train and X_test
    '''
    cat_cols = config['cols']
    k = config['dummy']['k'][0]
    
    logger.info('begin to transform')
    logger.info('begin to impute')
    #initializing impute_table
    imputer = Imputer(config['impute_table_name'])
    imputer.feed_impute_table(X_train, low_dt, high_dt, cat_cols)
        
    #imputing  
    X_train, X_test = imputer.impute_cat(X_train, X_test, cat_cols)
    X_train, X_test = imputer.impute_cond(X_train, X_test, cat_cols)
       
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
