############################################
# This script evaluate a trained model using the data with P value (i.e., probable the target) and using the whole
# training data.
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
    parser.add_argument('--input_model', type=str, default='model_data/gradient_boosting_2_labels_model.joblib',
                        help='saved model')
    parser.add_argument('--hsr_target', action='store_true',
                        help='Whether to add hsr_classification into target as well to make the model do two label '
                             'classification to classify both ecDNA and HSR to differentiate them')
    parser.add_argument('--output_data', type=str, default='data/CCLE_Mitelman_gradient_boosting_2_labels_p_predicted.csv',
                        help='output csv data')

    args = parser.parse_args()
    ori_input_data = args.ori_input_data
    input_data = args.input_data
    input_model = args.input_model
    hsr_target = args.hsr_target
    output_data = args.output_data

    target_columns = ['target', 'HSR_classification'] if hsr_target else ['target']
    predicted_target_columns = ['predicted_ecDNA_classification', 'predicted_HSR_classification']
    print(target_columns)
    _, _, df_1, input_df = read_and_process_data(input_data, target_columns=target_columns)
    print(df_1.shape, input_df.shape)
    for df in (df_1, input_df):
        # separate feature variables from the target variable
        X = df.drop(columns=target_columns, axis=1)  # features
        y = df[target_columns] # target
        # Initialize and scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # evaluate the model on the whole data
        classifier = joblib.load(input_model)
        y_pred = classifier.predict(X_scaled)
        # y_pred_prob = classifier.predict_proba(X_scaled)
        if df is df_1:
            print(y_pred)
            ori_df = pd.read_csv(ori_input_data)
            ori_df_p = ori_df[ori_df['ECDNA_classification'] == 'P']
            print(df.shape, ori_df_p.shape)
            if hsr_target:
                pred_df = pd.DataFrame(y_pred, columns=predicted_target_columns)
                ori_df_p.reset_index(inplace=True, drop=True)
                ori_df_p_y = pd.concat([ori_df_p, pred_df], axis=1)
                ori_df_p_y.to_csv(output_data, index=False)
            else:
                indices = np.where(y_pred == 1)[0]
                ori_df_p_1 = ori_df_p.iloc[indices]
                print(indices)
                ori_df_p_1.to_csv(output_data, index=False)
        else:
            accuracy, prec, recall, f1, ham_loss = evaluate_pred(y, y_pred, multilabel=hsr_target)
            if hsr_target:
                print(f"Accuracy: {accuracy}, precision: {prec}, recall: {recall}, f1: {f1}, hamming_loss: {ham_loss}")
            else:
                print(f"Accuracy: {accuracy:.2f}, precision: {prec: .2f}, recall: {recall: .2f}, f1: {f1: .2f}")
