"""
Imputation
"""
import logging
import sys
import pandas as pd
import numpy as np
import dbinterface

logger = logging.getLogger('start to transform the data')
ch = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.INFO)


class Imputer():
    """
    """
    
    def __init__(self, impute_table_name='impute_table'):
        '''
        attribute:
            cat_fill_value: the value to fill in categorical features
            dbinstance: the instance to connect with data base
            view_name: the name of view to fetch group by mean
        '''
        self.cat_fill_value = 'unknown'
        self.dbsource = dbinterface.DataSource()
        self.impute_table_name = impute_table_name
        # read build_impute_table.sql and build table
        try:
            with open('build_impute_table.sql') as f:
                sql_statement = f.read()
        except IOError:
            print("Cannot find 'build_impute_table.sql', please check it.")
            exit(2)
        sql_statement.replace('impute_table', impute_table_name)
        self.dbsource.dbinstance.execute_sql(sql_statement, True)
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
        #cond_cols.remove('transactiondt')
        #cond_cols.remove('transactionid')
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
        select_statement = '''SELECT {} FROM {} WHERE card4 = '{}' AND
            card6 = '{}';'''.format(column, self.impute_table_name, credit_type, card_type)
        cur = self.dbsource.dbinstance.execute_sql(select_statement)
        fetch_result = cur.fetchall()
        print(fetch_result)
        if fetch_result == []:
            print(select_statement)
            final_mean = -999
        else:
            final_mean = fetch_result[0][0]
        return final_mean
    
    
    def feed_impute_table(self, X_train, low_dt, high_dt, cat_cols):
        '''
        '''
        clear_cur = self.dbsource.dbinstance.execute_sql("DELETE FROM {}".format(self.impute_table_name))
        #card4_stat = "SELECT card4 from train_transaction group by card4"
        #card6_stat = "SELECT card6 from train_transaci group by card6"
        #card4_values = self.dbsource.dbinstance.execute_sql(card4_stat).fetchall()
        #card6_values = self.sbsource.dbinstance.execute_sql(card6_stat).fetchall()
        
        cond_cols = list(set(list(X_train.columns)) - set(cat_cols))
        #cond_cols.remove('transactiondt')
        #cond_cols.remove('transactionid')
        for column in cond_cols:
            cur = self.dbsource.compute_average_value('train', column, ['card4', 'card6'], low_dt, high_dt)
            fetch_results = cur.fetchall()
            for result in fetch_results:
                print(result)
                (credit_type, card_type, impute_value) = result
                if impute_value is None:
                    if X_train[column].mean() is not np.nan:
                        impute_value = X_train[column].mean()
                    else:
                        impute_value = -9999
                
                self.dbsource.set_impute_table_value(self.impute_table_name, credit_type, card_type, column, impute_value)
        
