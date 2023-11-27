import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.data import Data, InMemoryDataset
import torch_geometric
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.loader import DataLoader

from utils import load_data


class SimpleGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.fc = nn.Linear(output_dim, 3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_add_pool(x, data.batch)
        return self.fc(x)


class CustomGraphDataset(InMemoryDataset):
    def __init__(self, data_list, Y):
        super(CustomGraphDataset, self).__init__(".")
        self.data_list = data_list
        self.y = Y
        self._process()
        self.data, self.slices = self.collate(self.results)

    def _process_adjacency_matrix(self, adjacency_matrix, y):
        edge_index = adjacency_matrix.nonzero().t()
        edge_index = torch.stack((edge_index[0], edge_index[1]))
        node_features = torch.arange(0, 90, 1, dtype=torch.float)  # Replace with your node features
        node_features = node_features.unsqueeze(1)
        return Data(x=node_features, edge_index=edge_index, y=y)

    def _process(self):
        self.results = []
        for i, adj_matrix in enumerate(X):
            graph_data = self._process_adjacency_matrix(torch.tensor(adj_matrix), self.y[i])
            self.results.append(graph_data)

# Creating an instance of the custom dataset


if __name__ == '__main__':
    X, y = load_data()

    custom_dataset = CustomGraphDataset(data_list=X, Y=y)  # Pass your actual data_list here

    # Create edge indices from the adjacency matrix
    edge_index = torch.tensor(X[0]).nonzero().t()
    # Convert edge indices to a COO format
    edge_index = torch.stack((edge_index[0], edge_index[1]))

    # Node features (you can define your own node features)
    num_nodes = torch.tensor(X[0]).size(0)
    node_features = torch.range(0, 89, 1)  # Random node features for demonstration
    node_features = node_features.unsqueeze(1)
    # Creating a PyTorch Geometric Data object
    data = Data(x=node_features, edge_index=edge_index)

    g = torch_geometric.utils.to_networkx(data, to_undirected=True)
    nx.draw(g)
    plt.show()

    # Split the dataset into train and test (for demonstration purposes, adjust as needed)
    num_data = len(custom_dataset)
    num_train = int(0.8 * num_data)  # 80% train, 20% test
    num_test = num_data - num_train

    train_dataset, test_dataset = torch.utils.data.random_split(custom_dataset, [num_train, num_test])

    # Define DataLoader for batching
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Initialize the model
    model = SimpleGNN(input_dim=1, hidden_dim=20, output_dim=30)  # Adjust dimensions as needed

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001)

    # Training loop
    num_epochs = 50
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cpu"
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for data in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False):
            data = data.to(device)
            data.y = torch.tensor(data.y, dtype=torch.long)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y)  # Assuming the dataset has graph labels (data.y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.num_graphs

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}")

        # Evaluate on the test set
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data in tqdm(test_loader, desc='Testing', leave=False):
                data = data.to(device)
                data.y = torch.tensor(data.y, dtype=torch.long)
                output = model(data)
                _, predicted = torch.max(output, 1)
                total += data.y.size(0)
                correct += (predicted == data.y).sum().item()
                #print(output, predicted)

        accuracy = correct / total * 100
        print(f"Test Accuracy: {accuracy:.2f}%")
