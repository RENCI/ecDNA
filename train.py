############################################
# This script train a decision tree model using ec_master_imputed.csv data.
# It does feature scaling first before training to make sure all features
# have the same scale.
############################################
import time
import joblib
import pandas as pd
import argparse
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_pred(truth, pred):
    acc = accuracy_score(truth, pred)
    prec = precision_score(truth, pred)
    rec = recall_score(truth, pred)
    f1 = f1_score(truth, pred)
    return acc, prec, rec, f1


def read_and_process_data(filename):
    df = pd.read_csv(filename)
    df_1 = df[df.target == 1]
    # remove 'P' or 1 target rows from training data
    df = df[df.target != 1]
    df['target'].replace(2, 1, inplace=True)
    print(f'target values: {df.target.unique()}')
    # separate feature variables from the target variable
    feature = df.drop('target', axis=1)  # features
    target = df['target']
    return feature, target, df_1, df


def under_sample(f, t):
    rus = RandomUnderSampler(random_state=42)
    f_under, t_under = rus.fit_resample(f, t)
    return f_under, t_under


def scale_feature(f, t):
    f_train, f_test, t_train, t_test = train_test_split(f, t,
                                                        test_size=0.2, random_state=42)
    # Initialize and scale training and test data
    scaler = StandardScaler()
    f_train_scaled = scaler.fit_transform(f_train)
    f_test_scaled = scaler.transform(f_test)
    return f_train, f_test, t_train, t_test, f_train_scaled, f_test_scaled


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--input_data', type=str, default='data/ec_master_imputed.csv', help='input csv data')
    parser.add_argument('--model_type', type=str, default='random_forest')
    parser.add_argument('--output_model', type=str, default='model_data/random_forest_model.joblib',
                        help='saved model')

    args = parser.parse_args()
    input_data = args.input_data
    model_type = args.model_type
    output_model = args.output_model

    X, y, _, input_df = read_and_process_data(input_data)

    X_resampled, y_resampled = under_sample(X, y)

    ts = time.time()
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = scale_feature(X_resampled, y_resampled)

    if model_type == 'decision_tree':
        # train a decision tree classifier using the scaled features
        classifier = DecisionTreeClassifier(random_state=42)
    elif model_type == 'random_forest':
        # n_estimators set between 150 to 170 result in best overral performance metric
        classifier = RandomForestClassifier(n_estimators=150, random_state=42)
    elif model_type == 'svm':
        classifier = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    elif model_type == 'gradient_boosting':
        classifier = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42)
    else:
        print(f'model {model_type} is not supported, exiting')
        exit(1)

    classifier.fit(X_train_scaled, y_train)
    joblib.dump(classifier, output_model)

    if model_type == 'random_forest' or model_type == 'gradient_boosting':
        feat_imp = classifier.feature_importances_
        for i, importance in enumerate(feat_imp):
            if importance > 0:
                print(f"Feature {i} - {input_df.columns[i]}: {importance:.4f}")

    # evaluate the model
    y_pred = classifier.predict(X_test_scaled)
    print(y_pred)

    accuracy, precision, recall, f1 = evaluate_pred(y_test, y_pred)
    te = time.time()
    print(f"Accuracy: {accuracy:.2f}, precision: {precision: .2f}, recall: {recall: .2f}, f1: {f1: .2f},"
          f" time taken: {te-ts}")
