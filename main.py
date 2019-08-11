"""
"""
import sys
import argparse
import os
import gc
import yaml
import pandas as pd
import numpy as np
import dbinterface
import pickle
from pipeline import preprocess
from pipeline import evaluation
from pipeline import model_factory
import logging

logger = logging.getLogger('start to transform the data')
ch = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.INFO)

def run(config):
    '''
    '''
    logger.info("starting to run the pipeline")
    config = args.config
    with open (config) as config_file:
        configs = yaml.safe_load(config_file)
    start_time = configs['time']['start_time']
    end_time = configs['time']['end_time']
    interval = configs['time']['interval']
    trans_configs = configs['transform']
    model_configs = configs['models']
    matrix_configs = configs['matrix']
    if 'output_pred_probs_path' in matrix_configs:
        output_pred_probs_path = matrix_configs['output_pred_probs_path']
    else:
        output_pred_probs_path = None
    
    #split
    count = 1
    dbsource = dbinterface.DataSource()
    while True:
        #split train
        high_train = start_time + interval * count
        train_set_cur = dbsource.select_data_by_transactiondt('train', start_time, high_train, feature_list=None)
        #split test
        low_test = high_train + interval
        high_test = low_test + interval
        if end_time - high_test < interval:
            test_set_cur = dbsource.select_data_by_transactiondt('train', low_test, end_time, feature_list=None)
            break
        else:
            test_set_cur = dbsource.select_data_by_transactiondt('train', low_test, high_test, feature_list=None)
       
        train_set = dbinterface.cursor_to_dataframe(train_set_cur)
        test_set = dbinterface.cursor_to_dataframe(test_set_cur)
        X_train = train_set.drop(columns=['isfraud', 'transactionid', 'transactiondt'])
        X_test = test_set.drop(columns=['isfraud', 'transactionid', 'transactiondt'])
        y_train = train_set['isfraud']
        y_test = test_set['isfraud']
    
        #preprocessing
        X_train, X_test = preprocess.transform(trans_configs, X_train, X_test, start_time, high_train)
        
        #initializing result df
        results_df = pd.DataFrame(columns=matrix_configs['col_list'])
        #modeling
        for name, model in model_factory.get_models(model_configs):
                logger.info('start to run the model {}'.format(model))
                model.fit(X_train, y_train)
                print(sys.getsizeof(model))
                if name == 'LinearSVC':
                    y_pred_probs = model.decision_function(X_test)
                else:
                    y_pred_probs = model.predict_proba(X_test)[:, 1]
                if output_pred_probs_path is not None:
                    with open(os.path.join(output_pred_probs_path, name + '.pkl'), 'wb') as f:
                        pickle.dump(y_pred_probs, f)
                
                index = len(results_df)
                results_df.loc[index] = evaluation.get_matrix(results_df, y_pred_probs, y_test, name, model, count,index, matrix_configs)
                gc.collect()
                graph_name_roc = matrix_configs['roc_path'] + r'''roc_curve__{}_{}_{}'''.format(name,count,index)
                evaluation.plot_roc(str(model), graph_name_roc, y_pred_probs, y_test, 'save')
                
        results_df.to_csv(matrix_configs['out_path'] + str(count) + ".csv")
        count += 1
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Do a simple machine learning pipeline, load data, split the data, transform data, build                                       models, run models, get the performace matix results""")
    parser.add_argument('--config', dest='config', help='config file for this run', default ='./test_simple.yml')
    args = parser.parse_args()
    run(args)
        
