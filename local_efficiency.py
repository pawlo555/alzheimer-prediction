import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import load_data


if __name__ == '__main__':
    X, y = load_data()

    local_efficiencies = []
    global_efficiencies = []
    for i in tqdm(range(X.shape[0])):
        # Create a graph from the adjacency matrix
        G = nx.from_numpy_array(X[i])
        # print(str(i))
        local_efficiencies.append(nx.local_efficiency(G))
        global_efficiencies.append(nx.global_efficiency(G))

    local_efficiencies = np.array(local_efficiencies)
    global_efficiencies = np.array(global_efficiencies)

    plt.hist(local_efficiencies)
    plt.title("Local efficiency histogram")
    plt.show()
    plt.hist(global_efficiencies)
    plt.title("Global efficiency histogram")
    plt.show()

    labels = ["AD", "EMCI", "NC"]

    hists = []
    for i in range(3):
        hists.append(local_efficiencies[y == i])
    plt.title("Local efficiency histogram per class")
    plt.hist(hists, label=labels)
    plt.legend()
    plt.show()

    hists = []
    for i in range(3):
        hists.append(global_efficiencies[y == i])
    plt.title("Global efficiency histogram per class")
    plt.hist(hists, label=labels)
    plt.legend()
    plt.show()


# Local efficiency in per nodes wit networkx
# Apply this results in predicting alzheimer
# Compare groups for each pair based on global efficiency
# AD vs EMCI and EMCI vs NC ect.
# Based on local efficiency we can perform feature importance
# boxplots for global efficiency