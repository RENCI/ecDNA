############################################
# This script uses a trained classifer model to make inference on new data beyond training and validation data.
############################################
import joblib
import shap
import sys
import argparse
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
from src.train import read_and_process_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--train_data', type=str, default='../data/CCLE_Mitelman_for_ML_imputed.csv',
                        help='input training csv data')
    parser.add_argument('--input_model', type=str, default='../model_data/gradient_boosting_model.joblib',
                        help='saved model')
    parser.add_argument('--method', type=str, default='SHAP',
                        help='analysis method. Supported methods are PDP (Partial Dependency Plots) and '
                             'SHAP (SHapley Additive exPlanations)')
    parser.add_argument('--imp_features', type=list, default=['modal_range_numeric', 'HSR_classification',
                                                              'ploidy_classification'],
                        help='important feature list')
    parser.add_argument('--hsr_target', action='store_true',
                        help='Whether to add hsr_classification into target as well to make the model do two label '
                             'classification to classify both ecDNA and HSR to differentiate them')

    args = parser.parse_args()
    train_data = args.train_data
    input_model = args.input_model
    method = args.method
    imp_features = args.imp_features
    hsr_target = args.hsr_target

    target_columns = ['target', 'HSR_classification'] if hsr_target else ['target']

    X, _, _, input_df = read_and_process_data(train_data, target_columns=target_columns)
    train_features = input_df.drop(columns=target_columns)
    # make inference of the model on train data
    classifier = joblib.load(input_model)

    if method == 'PDP':
        feat_list = train_features.columns.tolist()
        print(feat_list)
        features = [feat_list.index(feat) for feat in imp_features]
        print(features)
        # Partial Dependence Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        PartialDependenceDisplay.from_estimator(classifier, X, features, ax=ax)
        plt.show()
    elif method == 'SHAP':
        explainer = shap.Explainer(classifier)
        shap_values = explainer(X)
        shap.initjs()
        shap.summary_plot(shap_values, X)
    else:
        print(f'input {method} is not supported')
        sys.exit(1)
    sys.exit(0)
