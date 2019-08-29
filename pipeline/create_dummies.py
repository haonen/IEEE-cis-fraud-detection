'''
Get the dummies in the features generation stage
'''

import numpy as np
import pandas as pd
import logging
import sys
import os
import gc

logger = logging.getLogger('get dummy')
ch = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.INFO)


class DummyFiller:
    def __init__(self, colname):
        self.colname = colname

    def fill(self, df):
        raise NotImplementedError()

class AllDummyFiller(DummyFiller):
    def __init__(self, colname, X_train):
        super().__init__(colname)
        self.cat_list = list(X_train[colname].value_counts().index.values)
    def fill(self, df):
        colname = self.colname
        for cat in self.cat_list:
            new_col_name = colname + "::" + cat
            df[new_col_name] = np.where(df[colname] == cat, 1, 0)

class TopKDummyFiller(DummyFiller):
    def __init__(self, colname, X_train, k):
        super().__init__(colname)
        self.top_k = X_train[colname].value_counts()[:k].index
    def fill(self, df):
        colname = self.colname
        top_k = self.top_k
        for cat in top_k:
            new_col_name = colname + "::" + cat
            df[new_col_name] = np.where(df[colname] == cat, 1, 0)
        df['{}_others'.format(colname)] = df.apply(
                lambda x: 0 if x[colname] in top_k else 1, axis=1)

def get_all_dummies(X_train, X_test, colname):
    '''
    Convert the categorical variable into dummies
    Inputs:
        X_train: a data frame of training set
        X_test: a data frame of test set (optional)
        colname: the name of the colname
    Return:
        the data frame with those dummies into data frame
        A DummyFiller will be returned for further use
    '''
    # Get the categories from training data set
    # cat_list = list(X_train[colname].value_counts().index.values)
    # create dummies
    # logger.info('generate the dummy on column {}'.format(colname))
    dummyfiller = AllDummyFiller(colname, X_train)
    dummyfiller.fill(X_train)
    if X_test is not None:
        dummyfiller.fill(X_test)
    return dummyfiller

def get_top_k_dummies(X_train, X_test, colname, k):
    '''
    For columns with too many categories, only create dummies for
    top k categories
    Inputs:
        X_train: a data frame of training set
        X_test: a data frame of test set (optional)
        colname: the name of the column
        k: (int) the value of k
    Outputs:
       Create dummies in both train and test set
    '''
    # get top k categories from train set
    # top_k = X_train[colname].value_counts()[:k].index
    dummyfiller = TopKDummyFiller(colname, X_train, k)
    # create dummies
    #logger.info('generate dummies on column {} of top {} values'.format(colname, k))
    dummyfiller.fill(X_train)
    if X_test is not None:
        dummyfiller.fill(X_test)
    return dummyfiller


def get_dummies(X_train, X_test, columns, k):
    '''
    Wrap up get_all_dummies and get_top_k_dummies
    Inputs:
        X_train: a data frame of training set
        X_test: a data frame of test set
        colname: the name of the column
        k: (int) the value of k
    Outputs:
       Create dummies in both train and test set
    '''
    # Decide whether this use get all dummies or top k
    for colname in columns:
        logger.info("get dummy for {}".format(colname))
        num_cat = len(X_train[colname].value_counts())

        if num_cat <= k or colname=='zip code':
            get_all_dummies(X_train, X_test, colname)
        else:
            get_top_k_dummies(X_train, X_test, colname, k)          
    return X_train.drop(columns=columns), X_test.drop(columns=columns)

def get_dummies_of_train(X_train, columns, k):
    '''
    Wrap up get_all_dummies and get_top_k_dummies
    Inputs:
        X_train: a data frame
        colname: the name of the column
        k: (int) the value of k
    Outputs:
       Create dummies in the data frame and a dict of dummy fillers
    '''
    # Decide whether this use get all dummies or top k
    dummyfillers = {}
    for colname in columns:
        logger.info("get dummy for {}".format(colname))
        num_cat = len(X_train[colname].value_counts())

        if num_cat <= k or colname=='zip code':
            dummyfillers[colname] = get_all_dummies(X_train, None, colname)
        else:
            dummyfillers[colname] = get_top_k_dummies(X_train, None, colname, k)

    return X_train.drop(columns=columns), dummyfillers

def get_dummies_of_test(X_test, columns, dummyfillers):
    '''
    Create dummy for test data
    Inputs:
        X_test: a data frame
        colname: the name of the column
        dummyfillers: the dict of dummy fillers
    Outputs:
       Create dummies in the data frame
    '''
    for colname in columns:
        logger.info("get dummy for {}".format(colname))
        dummyfillers[colname].fill(X_test)        
    return X_test.drop(columns=columns)


