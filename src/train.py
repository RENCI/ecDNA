############################################
# This script train a model using imputed data generated from impute_features.py.
# It does feature scaling first before training to make sure all features
# have the same scale.
############################################
import joblib
import os
import pandas as pd
import argparse
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, confusion_matrix, \
    multilabel_confusion_matrix, classification_report


def evaluate_pred(truth, pred, multilabel=False):
    average_method = 'binary'
    if multilabel:
        average_method = None
    acc = accuracy_score(truth, pred)
    prec = precision_score(truth, pred, average=average_method)
    rec = recall_score(truth, pred, average=average_method)
    f1 = f1_score(truth, pred, average=average_method)
    if multilabel:
        hloss = hamming_loss(truth, pred)
    else:
        hloss = None
    return acc, prec, rec, f1, hloss


def read_and_process_data(filename, target_columns=['target']):
    if 'target' not in target_columns:
        print("'target' must be in target_columns input parameter, exiting.")
        exit(1)
    df = pd.read_csv(filename)
    df_1 = df[df.target == 1]
    # remove 'P' or 1 target rows from training data
    df = df[df.target != 1]
    df['target'].replace(2, 1, inplace=True)
    print(f'target values: {df.target.unique()}, {len(df[df.target == 1])} target is 1, '
          f'{len(df[df.target == 0])} target is 0, HSR_classification values: {df.HSR_classification.unique()}, '
          f'{len(df[df.HSR_classification == 1])} target is 1, {len(df[df.HSR_classification == 0])} target is 0')
    # separate feature variables from the target variable
    feature = df.drop(columns=target_columns, axis=1)  # features
    target = df[target_columns]
    return feature, target, df_1, df


def scale_feature(f, t):
    f_train, f_test, t_train, t_test = train_test_split(f, t,
                                                        test_size=0.2, random_state=42)
    # Initialize and scale training and test data
    scaler = StandardScaler()
    f_train_scaled = scaler.fit_transform(f_train)
    f_test_scaled = scaler.transform(f_test)
    return f_train, f_test, t_train, t_test, f_train_scaled, f_test_scaled


def get_importance_features(model, feat_cols):
    imp_dict = {}
    for i, importance in enumerate(model.feature_importances_):
        if importance > 0:
            imp_dict[feat_cols[i]] = importance
    sorted_imp = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)
    return pd.DataFrame(sorted_imp, columns=['Feature', 'Importance'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--input_data', type=str, default='../data/CCLE_Mitelman_for_ML_imputed.csv', help='input csv data')
    parser.add_argument('--model_type', type=str, default='Gradient Boosting')
    parser.add_argument('--hsr_target', action='store_false',
                        help='Whether to add hsr_classification into target as well to make the model do two label '
                             'classification to classify both ecDNA and HSR to differentiate them')
    parser.add_argument('--output_model', type=str, default='../model_data/gradient_boosting_2_labels_model.joblib',
                        help='saved model')
    parser.add_argument('--output_analysis_data', type=str,
                        default='../results/analysis_data/gradient_boosting_features.csv',
                        help='sorted features with importance scores which are contributed to the classifier')

    args = parser.parse_args()
    input_data = args.input_data
    model_type = args.model_type
    hsr_target = args.hsr_target
    output_model = args.output_model
    output_analysis_data = args.output_analysis_data

    target_columns = ['target', 'HSR_classification'] if hsr_target else ['target']
    disp_target_columns = ['ecDNA', 'HSR']
    X, y, _, input_df = read_and_process_data(input_data, target_columns=target_columns)

    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = scale_feature(X, y)

    if model_type == 'decision_tree':
        # train a decision tree classifier using the scaled features
        classifier = DecisionTreeClassifier(random_state=42)
    elif model_type == 'Random Forest':
        if hsr_target:
            classifier = MultiOutputClassifier(RandomForestClassifier(n_estimators=150, random_state=42))
        else:
            # n_estimators set between 150 to 170 result in best overral performance metric
            classifier = RandomForestClassifier(n_estimators=150, random_state=42)
    elif model_type == 'svm':
        classifier = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    elif model_type == 'Gradient Boosting':
        if hsr_target:
            classifier = MultiOutputClassifier(GradientBoostingClassifier(n_estimators=150,
                                                                          learning_rate=0.1,
                                                                          random_state=42))
        else:
            classifier = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42)
    else:
        print(f'model {model_type} is not supported, exiting')
        exit(1)

    classifier.fit(X_train_scaled, y_train)
    joblib.dump(classifier, output_model)

    if model_type == 'Random Forest' or model_type == 'Gradient Boosting':
        if hasattr(classifier, 'feature_importances_'):
            importance_df = get_importance_features(classifier, X.columns)
            importance_df.to_csv(output_analysis_data, index=False)
        else:
            for i, clf in enumerate(classifier.estimators_):
                importance_df = get_importance_features(clf, X.columns)
                base, ext = os.path.splitext(output_analysis_data)
                importance_df.to_csv(f'{base}_{target_columns[i]}{ext}', index=False)

    # evaluate the model
    y_pred = classifier.predict(X_test_scaled)
    print(f"Classification report: {classification_report(y_test, y_pred)}")
    accuracy, precision, recall, f1, ham_loss = evaluate_pred(y_test, y_pred, multilabel=hsr_target)
    if hsr_target:
        cms = multilabel_confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix: {cms}")
        print(f"Accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}, hamming_loss: {ham_loss}")
        for i, cm in enumerate(cms):
            ax = plt.subplot()
            sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt='d')
            ax.set_title(f'Confusion Matrix of {disp_target_columns[i]} for {model_type}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Truth')
            ax.xaxis.set_ticklabels([f'{disp_target_columns[i]} (N)', f'{disp_target_columns[i]} (Y)'])
            ax.yaxis.set_ticklabels([f'{disp_target_columns[i]} (N)', f'{disp_target_columns[i]} (Y)'])
            # plt.show()
            # plt.savefig(f'{model_type.replace(" ", "")}_cm_plot_2_labels_{disp_target_columns[i]}.pdf', format='pdf',
            #             bbox_inches='tight')

        y_probs_pred_1 = classifier.predict_proba(X_test_scaled)[0][:, 1]
        # Compute ROC curve and AUC for Random Forest
        fpr1, tpr1, _ = roc_curve(y_test['target'], y_probs_pred_1)
        roc_auc1 = auc(fpr1, tpr1)
        y_probs_pred_2 = classifier.predict_proba(X_test_scaled)[1][:, 1]
        # Compute ROC curve and AUC for Random Forest
        fpr2, tpr2, _ = roc_curve(y_test['HSR_classification'], y_probs_pred_2)
        roc_auc2 = auc(fpr2, tpr2)

        plt.figure(figsize=(10, 6))
        plt.plot(fpr1, tpr1, color='blue', lw=2, label='ROC Curve (ecDNA) (area = %0.2f)' % roc_auc1)
        plt.plot(fpr2, tpr2, color='green', lw=2, label='ROC Curve (HSR) (area = %0.2f)' % roc_auc2)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        # plt.show()
        # plt.savefig(f'{model_type.replace(" ", "")}_2_labels_roc_plot.pdf', format='pdf', bbox_inches='tight')
    else:
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix: {cm}")
        print(f"Accuracy: {accuracy:.2f}, precision: {precision: .2f}, recall: {recall: .2f}, f1: {f1: .2f}")
        ax = plt.subplot()
        sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt='d')
        ax.set_title(f'Confusion Matrix for {model_type}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Truth')
        ax.xaxis.set_ticklabels(['ecDNA (N)', 'ecDNA (Y)'])
        ax.yaxis.set_ticklabels(['ecDNA (N)', 'ecDNA (Y)'])
        # plt.show()
        plt.savefig(f'{model_type.replace(" ", "")}_cm_plot.pdf', format='pdf', bbox_inches='tight')
        y_probs_pred = classifier.predict_proba(X_test_scaled)[:, 1]
        # Compute ROC curve and AUC for Random Forest
        fpr, tpr, _ = roc_curve(y_test, y_probs_pred)
        roc_auc = auc(fpr, tpr)
        # Plot the ROC curves
        plt.figure(figsize=(10, 7))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_type} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        # plt.show()
        # plt.savefig(f'{model_type.replace(" ", "")}_roc_plot.pdf', format='pdf', bbox_inches='tight')

