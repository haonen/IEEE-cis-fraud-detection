import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def concat_df(path1, path2, path3):
    '''
    '''
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    df3 = pd.read_csv(path3)
    df1['validation'] = "1"
    df2['validation'] = "2"
    df3['validation'] = "3"
    new_df = pd.concat([df1, df2,df3])
    return new_df

def plot_performance(df, evaluation_mertics, combo):
    '''
    '''
    for metric in evaluation_mertics:
        fig, ax1 = plt.subplots(1,1)
        g = sns.lineplot(data=df,x='validation', y='auc_roc_30%', hue='model_name', units='parameters', lw=1, estimator=None, ax=ax1)
        box = g.get_position()
        g.set_position([box.x0, box.y0, box.width * 0.63, box.height]) 
        g.legend(loc='center right', bbox_to_anchor=(1.8, 0.5), ncol=1)
        #plt.title('{} of {} across different validation'.format(metric, combo))
        #plt.savefig('{} of {} across different validation'.format(metric, combo))
        