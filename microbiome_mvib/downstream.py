"""
Downstream classifiers for the embeddings.
"""
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np


def get_metrics(clf, y_true, y_prob, y_pred):
    """
    Compute a dictionary with all classification metrics.
    """
    metrics = {
        'cross_val_best_params': clf.best_params_,
        'cross_val_best_score': clf.best_score_,
        'test_roc_auc_score': round(roc_auc_score(y_true, y_prob[:, 1]), 4),
        'test_accuracy_score': round(accuracy_score(y_true, y_pred), 4),
        'test_recall_score': round(recall_score(y_true, y_pred), 4),
        'test_precision_score': round(precision_score(y_true, y_pred), 4),
        'test_f1_score': round(f1_score(y_true, y_pred), 4),
        'test_sum_score': round(
            roc_auc_score(y_true, y_prob[:, 1]) +
            accuracy_score(y_true, y_pred) +
            recall_score(y_true, y_pred) +
            precision_score(y_true, y_pred) +
            f1_score(y_true, y_pred),
            4
        )
    }
    return metrics


def classification(X_train, X_test, y_train, y_test, args, models):
    """
    For each classifier in `models` operate a grid-search over the hyper-parameters space.
    """
    # scale features
    scaler = StandardScaler()
    scaler.fit(np.concatenate((X_train, X_test), axis=0))
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # store metrics for all downstream models
    metrics = dict()
    val_scores = dict()
    test_scores = dict()

    if 'svm' in models:
        # SVM
        svm_hyper_parameters = [
            {'C': [2 ** s for s in range(-5, 6, 2)], 'kernel': ['linear']},
            {'C': [2 ** s for s in range(-5, 6, 2)], 'gamma': [2 ** s for s in range(3, -15, -2)], 'kernel': ['rbf']}
        ]
        clf = GridSearchCV(
            SVC(probability=True, cache_size=10000),
            svm_hyper_parameters,
            cv=StratifiedKFold(n_splits=5, shuffle=True),
            scoring='roc_auc',
            n_jobs=args.n_jobs,
            verbose=100
        )
        clf.fit(X_train, y_train)

        y_true, y_pred = y_test, clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)

        svm_metrics = get_metrics(clf, y_true, y_prob, y_pred)
        metrics['svm'] = svm_metrics

    if 'rf' in models:
        # Random Forest
        rf_hyper_parameters = [
            {
                'n_estimators': [s for s in range(200, 1001, 200)],
                'max_features': ['sqrt', 'log2'],
                'min_samples_leaf': [1, 2, 3, 4, 5],
                'criterion': ['gini', 'entropy']
            }
        ]

        clf = GridSearchCV(
            RandomForestClassifier(),
            rf_hyper_parameters,
            cv=StratifiedKFold(n_splits=5, shuffle=True),
            scoring='roc_auc',
            n_jobs=args.n_jobs,
            verbose=100
        )
        clf.fit(X_train, y_train)

        y_true, y_pred = y_test, clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)

        rf_metrics = get_metrics(clf, y_true, y_prob, y_pred)
        metrics['rf'] = rf_metrics

    for model in models:
        val_scores[model] = metrics[model]['cross_val_best_score']
        test_scores[model] = metrics[model]['test_roc_auc_score']

    return metrics, val_scores, test_scores
