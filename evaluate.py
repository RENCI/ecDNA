############################################
# This script train a decision tree model using ec_master_imputed.csv data.
# It does feature scaling first before training to make sure all features
# have the same scale.
############################################
import joblib
import argparse
from sklearn.preprocessing import StandardScaler
from train import evaluate_pred, read_and_process_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--input_data', type=str, default='data/ec_master_imputed.csv', help='input csv data')
    parser.add_argument('--input_model', type=str, default='model_data/ensemble_model.joblib',
                        help='saved decision tree model')

    args = parser.parse_args()
    input_data = args.input_data
    input_model = args.input_model

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
            print(y_pred_prob.shape)
            print(y_pred_prob, y_pred)
        else:
            accuracy, prec, recall, f1 = evaluate_pred(y, y_pred)
            print(f"Accuracy: {accuracy:.2f}, precision: {prec: .2f}, recall: {recall: .2f}, f1: {f1: .2f}")
