import pandas as pd
import numpy as np

def submit(input_path, output_path):
    '''
    '''
    input_df = pd.read_csv(input_path)
    input_df.columns = ["TransactionID", "IsFraud"]
    input_df = input_df.sort_values(by=['TransactionID'])
    input_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    submit("./output/rf/final_predict_score.csv", "./output/rf/final_predict_score(submission).csv")
        