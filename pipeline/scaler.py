import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import MinMaxScaler


def min_max_transformation(train_df, test_df, continuous_columns):
    '''
    This function is used to perform min-max scaling for all variables in the
    training data, and transform both training and testing dataframe
    Inputs:
        train_df: training dataframe
        test_df: testing dataframe
    Returns: updated dataframes
    '''
    scaler = MinMaxScaler()
    train_df_cont = scaler.fit_transform(train_df[continuous_columns])
    test_df_cont = scaler.transform(test_df[continuous_columns])
    print(len(train_df_cont))
    print(len(test_df_cont))
    train_df = train_df.drop(continuous_columns, axis=1).reset_index(drop=True)
    test_df = test_df.drop(continuous_columns, axis=1).reset_index(drop=True)
    train_df = train_df.join(pd.DataFrame(data=train_df_cont, columns=continuous_columns))
    test_df = test_df.join(pd.DataFrame(data=test_df_cont,columns=continuous_columns))
    gc.collect()
    return train_df, test_df

def min_max_transformation_train(train_df, continuous_columns):
    '''
    This function is used to perform min-max scaling for all variables in the
    training data, and transform both training and testing dataframe
    Inputs:
        train_df: training dataframe
        test_df: testing dataframe
    Returns: updated dataframes
    '''
    scaler = MinMaxScaler()
    train_df_cont = scaler.fit_transform(train_df[continuous_columns])
    print(len(train_df_cont))
    train_df = train_df.drop(continuous_columns, axis=1).reset_index(drop=True)
    train_df = train_df.join(pd.DataFrame(data=train_df_cont, columns=continuous_columns))
    return train_df, scaler

def min_max_transformation_test(test_df, scaler, continuous_columns):
    '''
    This function is used to perform min-max scaling for all variables in the
    training data, and transform both training and testing dataframe
    Inputs:
        train_df: training dataframe
        test_df: testing dataframe
    Returns: updated dataframes
    '''
    test_df_cont = scaler.transform(test_df[continuous_columns])
    print(len(test_df_cont))
    test_df = test_df.drop(continuous_columns, axis=1).reset_index(drop=True)
    test_df = test_df.join(pd.DataFrame(data=test_df_cont,columns=continuous_columns))
    return test_df

''
