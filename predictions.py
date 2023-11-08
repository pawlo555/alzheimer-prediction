import os
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import svm


def load_data():
    X, y = [], []
    for class_id, folder in enumerate(["baseline_AD_bin", "baseline_EMCI_bin", "baseline_NC_bin"]):
        path = os.path.join("Connectomes", folder)
        for filename in os.listdir(path):
            if filename.endswith('.csv'):
                data = pd.read_csv(os.path.join(path, filename), header=None)
                X.append(np.array(data))
                y.append(class_id)
    return np.array(X), np.array(y)


def classify(X, y):
    knn = KNeighborsClassifier()
    params = {
        "n_neighbors": [5, 9, 11, 13, 15],
        "weights": ["uniform", "distance"],
        "metric": ["minkowski", "cosine", "euclidean"]
    }
    X = np.reshape(X, (X.shape[0], -1))
    clf = GridSearchCV(knn, params, cv=5, scoring="accuracy")
    clf.fit(X, y)
    df = pd.DataFrame(columns=["Accuracy", "N_neighbors", "Weights", "Metric"])
    for i in range(len(clf.cv_results_["params"])):
        df.loc[i] = [clf.cv_results_["mean_test_score"][i],
                     clf.cv_results_["params"][i]["n_neighbors"],
                     clf.cv_results_["params"][i]["weights"],
                     clf.cv_results_["params"][i]["metric"]]
    df = df.sort_values(by="Accuracy", ascending=False)
    print(df.head())


if __name__ == '__main__':
    X, y = load_data()
    print(X.shape)
    print(y.shape)
    classify(X, y)
