############################################
# This script train a decision tree model using ec_master_imputed.csv data.
# It does feature scaling first before training to make sure all features
# have the same scale.
############################################
import joblib
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from train import evaluate_pred, read_and_process_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--ori_input_data', type=str, default='data/CCLE_Mitelman_for_ML.csv',
                        help='original input csv data')
    parser.add_argument('--input_data', type=str, default='data/CCLE_Mitelman_for_ML_imputed.csv', help='input csv data')
    parser.add_argument('--input_model', type=str, default='model_data/random_forest_model.joblib',
                        help='saved model')
    parser.add_argument('--output_data', type=str, default='data/CCLE_Mitelman_random_forest_p_predicted_y.csv',
                        help='output csv data')

    args = parser.parse_args()
    ori_input_data = args.ori_input_data
    input_data = args.input_data
    input_model = args.input_model
    output_data = args.output_data

    _, _, df_1, input_df = read_and_process_data(input_data)
    print(df_1.shape, input_df.shape)
    for df in (df_1, input_df):
        # separate feature variables from the target variable
        X = df.drop('target', axis=1) # features
        y = df['target'] # target
        # Initialize and scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # evaluate the model on the whole data
        classifier = joblib.load(input_model)
        y_pred = classifier.predict(X_scaled)
        y_pred_prob = classifier.predict_proba(X_scaled)
        if df is df_1:
            print(y_pred)
            indices = np.where(y_pred == 1)[0]
            print(indices)
            ori_df = pd.read_csv(ori_input_data)
            ori_df_p = ori_df[ori_df['ECDNA_classification'] == 'P']
            print(df.shape, ori_df_p.shape)
            ori_df_p_1 = ori_df_p.iloc[indices]
            print(ori_df_p_1['dataset'])
            ori_df_p_1.to_csv(output_data, index=False)
        else:
            accuracy, prec, recall, f1, _ = evaluate_pred(y, y_pred)
            print(f"Accuracy: {accuracy:.2f}, precision: {prec: .2f}, recall: {recall: .2f}, f1: {f1: .2f}")
