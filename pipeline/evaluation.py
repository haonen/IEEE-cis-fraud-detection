import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
import matplotlib.pyplot as plt


def compute_auc_roc(y_true, y_scores, k):
    '''
    Compute area under Receiver Operator Characteristic Curve
    :param pred_scores: (np array) an array of predicted score
    :param threshold: (float) the threshold of labeling predicted results
    :param y_test: test set
    :return: (float) an auc_roc score
    '''
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)

    return roc_auc_score(y_true_sorted, preds_at_k)


def compute_auc(pred_scores, true_labels):
    '''
    Compute auc score
    :param pred_scores: an array of predicted scores
    :param true_labels: an array of true labels
    :return: area under curve score
    '''
    fpr, tpr, thresholds = roc_curve(true_labels, pred_scores, pos_label=2)
    return auc(fpr, tpr)


def joint_sort_descending(l1, l2):
    '''
    Sort two arrays together
    :param l1:  numpy array
    :param l2:  numpy array
    :return: two sorted arrays
    '''
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]


def generate_binary_at_k(y_scores, k):
    '''
    predict labels based on thresholds
    :param y_scores: the predicted scores
    :param k: (int or float) threshold
    :return: predicted labels
    '''
    cutoff_index = int(len(y_scores) * (k / 100.0))
    cutoff_val = y_scores[cutoff_index]
    predictions_binary = [1 if x >= cutoff_val else 0 for x in y_scores]
    return predictions_binary


def plot_roc(name, save_name, probs, y_true, output_type):
    '''
    Plot auc_roc curve
    '''
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    plt.clf()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(name)
    plt.legend(loc="lower right")
    if (output_type == 'save'):
        plt.savefig(save_name, close=True)
        plt.close()
    elif (output_type == 'show'):
        plt.show()
    else:
        plt.show()
        
        
def get_matrix(results_df, y_pred_probs, y_test, name, model, count, index, matrix_configs):
    '''
    calculate the evaluation matrixs
 
    Input:
        results_df: used to store the result
        y_pred_probs: get the score from the model
        y_test: true y
        name: model's name
        model: model obj
        count: number of train test set
    Return:
        one row of record for the result dataframe 
    '''
    # Sort true y labels and predicted scores at the same time
    y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
    # Write the evaluation results into data frame
    threshold_list = matrix_configs['percentage']
    record = [name, str(model)]
    for t in threshold_list:
        record.append(compute_auc_roc(y_test_sorted, y_pred_probs_sorted, t))
 
    graph_name_roc = matrix_configs['roc_path'] + r'''roc_curve__{}_{}_{}'''.format(name,count,index)
    #plot_roc(str(model), graph_name_roc, y_pred_probs, y_test, 'save')
    return record
