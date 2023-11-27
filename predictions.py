import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

from utils import load_data


def classify(X: np.ndarray, y: np.ndarray, method: str = 'knn') -> None:
    classifiers = {
        'knn': (KNeighborsClassifier(), {
            "n_neighbors": [5, 9, 11, 13, 15],
            "weights": ["uniform", "distance"],
            "metric": ["minkowski", "cosine", "euclidean"]
        }),
        'svc': (SVC(), {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf", "poly"],
            "gamma": ["scale", "auto"],
        }),
        'random_forest': (RandomForestClassifier(), {
            "n_estimators": [50, 100, 150],
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 10, 20, 30]
        }),
        'xgboost' : (xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss'), {
            'max_depth': [5, 12, 20],
            'learning_rate': [0.5, 0.4, 0.3],
            'n_estimators': [100, 150, 200],
        }),
    }

    classifier, params = classifiers.get(method, (None, None))
    if classifier is None or params is None:
        raise ValueError(f'Classifier "{method}" is not implemented!')

    X = np.reshape(X, (X.shape[0], -1))
    clf = GridSearchCV(classifier, params, cv=5, scoring="accuracy")
    clf.fit(X, y)
    df = pd.DataFrame(columns=["Accuracy"] + list(params.keys()))
    for i in range(len(clf.cv_results_["params"])):
        df.loc[i] = [clf.cv_results_["mean_test_score"][i]] + [clf.cv_results_["params"][i][param] for param in params]
    df = df.sort_values(by="Accuracy", ascending=False)
    print(f'Results for {method}:')
    print(df.head(), end='\n\n')


if __name__ == '__main__':
    X, y = load_data()
    print(X.shape)
    print(y.shape)
    classify(X, y, 'knn')
    classify(X, y, 'svc')
    classify(X, y, 'random_forest')
    # Long calculations
    classify(X, y, 'xgboost')
    # END Long calculations
