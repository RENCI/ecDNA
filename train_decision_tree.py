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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_pred(truth, pred):
    acc = accuracy_score(truth, pred)
    prec = precision_score(truth, pred)
    rec = recall_score(truth, pred)
    f1 = f1_score(truth, pred)
    return acc, prec, rec, f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--input_data', type=str, default='data/ec_master_imputed.csv', help='input csv data')
    parser.add_argument('--output_model', type=str, default='model_data/decision_tree_model.joblib',
                        help='saved decision tree model')

    args = parser.parse_args()
    input_data = args.input_data
    output_model = args.output_model

    input_df = pd.read_csv(input_data)

    # remove 'P' or 1 target rows from training data
    input_df = input_df[input_df.target != 1]
    input_df['target'].replace(2, 1, inplace=True)
    print(f'target values: {input_df.target.unique()}')
    # separate feature variables from the target variable
    X = input_df.drop('target', axis=1)  # features
    y = input_df['target']  # target

    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X, y)

    ts = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled,
                                                        test_size=0.2, random_state=42)
    # Initialize and scale training and test data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # train a decision tree classifier using the scaled features
    classifier = DecisionTreeClassifier(random_state=42)
    classifier.fit(X_train_scaled, y_train)
    joblib.dump(classifier, output_model)
    # evaluate the model
    y_pred = classifier.predict(X_test_scaled)
    print(y_pred)

    accuracy, precision, recall, f1 = evaluate_pred(y_test, y_pred)
    te = time.time()
    print(f"Accuracy: {accuracy:.2f}, precision: {precision: .2f}, recall: {recall: .2f}, f1: {f1: .2f},"
          f" time taken: {te-ts}")
