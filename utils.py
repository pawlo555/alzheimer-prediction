import os
import pandas as pd
import numpy as np

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
