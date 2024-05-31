############################################
# This script uses a trained classifer model to make inference on new data beyond training and validation data.
############################################
import joblib
import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from convert_data_to_numerical import ploidy_mapping, chromosome_mapping, map_feature


def get_range_number(in_range):
    in_range = str(in_range)
    split_range = in_range.split('-')
    if len(split_range) == 1:
        return int(split_range[0])
    else:
        return int(split_range[1]) - int(split_range[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--input_data', type=str, default='../data/Mouse_karyotypes_ECDNA.csv', help='input csv data')
    parser.add_argument('--train_data', type=str, default='../data/CCLE_Mitelman_for_ML_imputed.csv',
                        help='input training csv data')
    parser.add_argument('--input_model', type=str, default='../model_data/gradient_boosting_model.joblib',
                        help='saved model')
    parser.add_argument('--hsr_target', action='store_true',
                        help='Whether to add hsr_classification into target as well to make the model do two label '
                             'classification to classify both ecDNA and HSR to differentiate them')
    parser.add_argument('--output_data', type=str,
                        default='../data/Mouse_karyotypes_gradient_boosting_predicted_2.csv',
                        help='output csv data')

    args = parser.parse_args()
    input_data = args.input_data
    train_data = args.train_data
    input_model = args.input_model
    hsr_target = args.hsr_target
    output_data = args.output_data

    input_df = pd.read_csv(input_data)
    input_features = input_df.drop(columns=['cellline'])  # features
    # change near-triploid to triploid before mapping since near-triploid is not in the mapping key
    # input_feature.loc[input_feature['ploidy_classification'] == 'near-triploid', 'ploidy_classification'] = 'triploid'
    # map ploidy_classification to numerical
    input_features['ploidy_classification'] = map_feature(input_features['ploidy_classification'], ploidy_mapping)
    # map XY_chromosomes to numerical
    input_features['XY_chromosomes'] = map_feature(input_features['XY_chromosomes'], chromosome_mapping)
    input_features['modal_range_numeric'] = input_features['modal_range_numeric'].apply(lambda x: get_range_number(x))

    train_df = pd.read_csv(train_data)
    if hsr_target:
        train_features = train_df.drop(columns=['HSR_classification', 'target'])
    else:
        train_features = train_df.drop(columns=['target'])
    missing_features = [col for col in train_features.columns if col not in input_features.columns]
    df_missing_features = pd.DataFrame(columns=missing_features)
    df_missing_features = df_missing_features.fillna(-1)
    input_features_all = pd.concat([input_features, df_missing_features], axis=1)
    input_features_all.fillna(-1, inplace=True)
    input_features_reordered = input_features_all[train_features.columns]

    # Initialize and scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(input_features_reordered)

    # make inference of the model on the whole data
    classifier = joblib.load(input_model)
    y_pred = classifier.predict(X_scaled)
    y_pred_prob = classifier.predict_proba(X_scaled)
    if hsr_target:
        Y_pred_df = pd.DataFrame(y_pred, columns=['ECDNA_classification', 'HSR_classification'])
    else:
        Y_pred_df = pd.DataFrame(y_pred, columns=['ECDNA_classification'])
    pd.concat([input_df, Y_pred_df], axis=1).to_csv(output_data, index=False)
    print(y_pred)
    print(y_pred_prob)
