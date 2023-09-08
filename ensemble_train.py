############################################
# This script train a decision tree model using ec_master_imputed.csv data.
# It does feature scaling first before training to make sure all features
# have the same scale.
############################################
import time
import joblib
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, StackingClassifier
from train import evaluate_pred, read_and_process_data, under_sample, scale_feature


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--input_data', type=str, default='data/ec_master_imputed.csv', help='input csv data')
    parser.add_argument('--output_model', type=str, default='model_data/ensemble_model.joblib',
                        help='saved model')

    args = parser.parse_args()
    input_data = args.input_data
    output_model = args.output_model

    X, y, _, _ = read_and_process_data(input_data)

    X_resampled, y_resampled = under_sample(X, y)

    ts = time.time()
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = scale_feature(X_resampled, y_resampled)
    base_estimator = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42)
    bagging_classifier = BaggingClassifier(estimator=base_estimator, n_estimators=100, random_state=42)
    bagging_classifier.fit(X_train_scaled, y_train)

    meta_estimator = RandomForestClassifier(n_estimators=150, random_state=42)

    estimators = [('bagging', bagging_classifier)]
    stacking_classifier = StackingClassifier(estimators=estimators, final_estimator=meta_estimator)
    stacking_classifier.fit(X_train_scaled, y_train)

    joblib.dump(stacking_classifier, output_model)
    # evaluate the model
    y_pred = stacking_classifier.predict(X_test_scaled)
    print(y_pred)

    accuracy, precision, recall, f1 = evaluate_pred(y_test, y_pred)
    te = time.time()
    print(f"Accuracy: {accuracy:.2f}, precision: {precision: .2f}, recall: {recall: .2f}, f1: {f1: .2f},"
          f" time taken: {te-ts}")
