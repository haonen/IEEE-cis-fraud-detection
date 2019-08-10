"""
Imputation
"""
import logging
import sys
import pandas as pd
import numpy as np
import dbinterface
from config import config

logger = logging.getLogger('start to transform the data')
ch = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.INFO)


class Imputer():
    """
    """
    
    def __init__(self):
        '''
        attribute:
            cat_fill_value: the value to fill in categorical features
            dbinstance: the instance to connect with data base
            view_name: the name of view to fetch group by mean
        '''
        self.cat_fill_value = 'unknown'
        self.dbinstance = dbinterface.DBInterface(config.USERNAME, config.PASSWORD, '127.0.0.1', '5432', 'ieeefraud')
        self.view_name = "to be set"
        #self.cat_cols = cat_cols
        
    def impute_cat(self, X_train, X_test, cat_cols):
        '''
        Impute categorical features
        Inputs:
            X_train: train df
            X_test: test df
            cat_cols: a list of catgorical column names
        Returns:
            X_train and X_test
        '''
        for column in cat_cols:
            logger.info('imputing {}'.format(column))
            X_train[column].fillna(self.cat_fill_value)     
            X_test[column].fillna(self.cat_fill_value)
        return X_train, X_test
    
    
    def impute_cond(self, X_train, X_test, cat_cols):
        '''
        Impute continuous features
        Inputs:
            X_train: train df
            X_test: test df
            cat_cols: a list of catgorical column names
        Returns:
            X_train and X_test
        '''
        cond_cols = list(set(list(X_train.columns)) - set(cat_cols))
        cat_card4 = list(X_train['card4'].value_counts().index)
        cat_card6 = list(X_train['card6'].value_counts().index)
        for column in cond_cols:
            logger.info('imputing {}'.format(column))
            for credit_type in cat_card4:
                for card_type in cat_card6:
                    group_mean = self.fetch_group_mean(column, credit_type, card_type)
                    
                    condition_train = ((X_train[column].isnull()) & (X_train['card4'] == credit_type) 
                                       & (X_train['card6'] == card_type))
                    X_train.loc[condition_train, column] = group_mean
                    condition_test = ((X_test[column].isnull()) & (X_test['card4'] == credit_type)
                                      & (X_test['card6'] == card_type))
                    X_test.loc[condition_test, column] = group_mean
                    
        return X_train, X_test
    
    
    def fetch_group_mean(self, column, credit_type, card_type):
        '''
        Fetch the aggregated mean of a certain type of credit card
        Inputs:
            column:(str) the name of the column to be imputed
            credit_type: (str) the name of the credit type
            card_type: (str) the name of the card type
        Returns:
            aggregated mean(float)
        '''
        select_statement = '''SELECT {} FROM {} WHERE card4 = {} AND
            card6 = {};'''.format(column, self.view_name, credit_type, card_type)
        self.dbinstance.cur.execute(select_statement)
        fetch_result = self.dbinstance.cur.fetchall()
        self.dbinstance.connection.commit()
        return fetch_result[0][0]
    