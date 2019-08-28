"""
"""
import sys
import argparse
import os
import gc
import yaml
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import dbinterface
import pickle
from pipeline import preprocess
from pipeline import evaluation
from pipeline import model_factory
import logging
from IPython import embed;

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
    trans_configs = configs['transform']
    model_configs = configs['models']
    matrix_configs = configs['matrix']
    if 'output_pred_probs_path' in matrix_configs:
        output_pred_probs_path = matrix_configs['output_pred_probs_path']
    else:
        output_pred_probs_path = None
    
    dbsource = dbinterface.DataSource()
  
    train_set_cur = dbsource.select_data_by_transactiondt('train', start_time, end_time, feature_list=None)
    train_set = dbinterface.cursor_to_dataframe(train_set_cur)
    test_set_cur = dbsource.select_all_data('test', feature_list=None)
    test_set = dbinterface.cursor_to_dataframe(test_set_cur)
    print("Train data count = {}".format(len(train_set)))
    print("Test data count = {}".format(len(test_set)))
    X_train = train_set.drop(columns=['isfraud', 'transactionid', 'transactiondt'])
    X_test = test_set.drop(columns=['transactionid', 'transactiondt'])
    y_train = train_set['isfraud']
    transactionid_test = test_set['transactionid']

    #preprocessing
    X_train, X_test = preprocess.transform(trans_configs, X_train, X_test, start_time, end_time)

    #initializing result dictionary
    results_dict = {}
    results_dict['transactionid'] = transactionid_test.to_numpy()[:, 0]
    #modeling
    model_id = 0
    for name, model in model_factory.get_models(model_configs):
        model_id += 1
        logger.info('start to run the model {}'.format(model))
        # check NaN and replace them with the birthday of the person
        X_train[X_train.isna()] = -0.19260817
        X_test[X_test.isna()] = -0.19260817
        model.fit(X_train, y_train)
        print(sys.getsizeof(model))
        if name in ['LinearSVC', 'SVC']:
            y_pred_probs = model.decision_function(X_test)
        else:
            y_pred_probs = model.predict_proba(X_test)[:, 1]
        if output_pred_probs_path is not None:
            with open(os.path.join(output_pred_probs_path, name + '.pkl'), 'wb') as f:
                pickle.dump(y_pred_probs, f)
        results_dict['model_{}'.format(model_id)] = y_pred_probs
        gc.collect()
    results_df = pd.DataFrame(data=results_dict)
    results_df.to_csv(matrix_configs['final_score_path'] + "final_predict_score.csv", index=False)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Do a simple machine learning pipeline, load data, split the data, transform data, build                                       models, run models, get the performace matix results""")
    parser.add_argument('--config', dest='config', help='config file for this run', default ='./test_simple.yml')
    args = parser.parse_args()
    run(args)
        
