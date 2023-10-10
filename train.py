############################################
# This script train a decision tree model using ec_master_imputed.csv data.
# It does feature scaling first before training to make sure all features
# have the same scale.
############################################
import time
import joblib
import pandas as pd
import argparse
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
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
    print(f'target values: {df.target.unique()}, {len(df[df.target == 1])} target is 1, '
          f'{len(df[df.target == 0])} target is 0')
    # separate feature variables from the target variable
    feature = df.drop('target', axis=1)  # features
    target = df['target']
    return feature, target, df_1, df


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
    parser.add_argument('--input_data', type=str, default='data/CCLE_Mitelman_for_ML_imputed.csv', help='input csv data')
    parser.add_argument('--model_type', type=str, default='Random Forest')
    parser.add_argument('--output_model', type=str, default='model_data/random_forest_model.joblib',
                        help='saved model')
    parser.add_argument('--output_analysis_data', type=str, default='analysis_data/random_forest_features.csv',
                        help='sorted features with importance scores which are contributed to the classifier')

    args = parser.parse_args()
    input_data = args.input_data
    model_type = args.model_type
    output_model = args.output_model
    output_analysis_data = args.output_analysis_data

    X, y, _, input_df = read_and_process_data(input_data)

    ts = time.time()
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = scale_feature(X, y)

    if model_type == 'decision_tree':
        # train a decision tree classifier using the scaled features
        classifier = DecisionTreeClassifier(random_state=42)
    elif model_type == 'Random Forest':
        # n_estimators set between 150 to 170 result in best overral performance metric
        classifier = RandomForestClassifier(n_estimators=150, random_state=42)
    elif model_type == 'svm':
        classifier = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    elif model_type == 'Gradient Boosting':
        classifier = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42)
    else:
        print(f'model {model_type} is not supported, exiting')
        exit(1)

    classifier.fit(X_train_scaled, y_train)
    joblib.dump(classifier, output_model)

    if model_type == 'Random Forest' or model_type == 'Gradient Boosting':
        importance_dict = {}
        for i, importance in enumerate(classifier.feature_importances_):
            if importance > 0:
                importance_dict[input_df.columns[i]] = importance
                print(f"Feature {i} - {input_df.columns[i]}: {importance:.4f}")
        sorted_importance = sorted(importance_dict.items(), key=lambda x:x[1], reverse=True)
        importance_df = pd.DataFrame(sorted_importance, columns=['Feature', 'Importance'])
        importance_df.to_csv(output_analysis_data, index=False)

    # evaluate the model
    y_pred = classifier.predict(X_test_scaled)
    print(y_pred)
    y_probs_pred = classifier.predict_proba(X_test_scaled)[:, 1]
    print(y_probs_pred)
    # Compute ROC curve and AUC for Random Forest
    fpr, tpr, _ = roc_curve(y_test, y_probs_pred)
    roc_auc = auc(fpr, tpr)

    accuracy, precision, recall, f1 = evaluate_pred(y_test, y_pred)
    te = time.time()
    print(f"Accuracy: {accuracy:.2f}, precision: {precision: .2f}, recall: {recall: .2f}, f1: {f1: .2f},"
          f" time taken: {te-ts}")
    # Plot the ROC curves
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_type} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    # plt.show()
    plt.savefig(f'{model_type.replace(" ", "")}_roc_plot.pdf', format='pdf', bbox_inches='tight')
