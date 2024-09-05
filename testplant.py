import os
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, ARGVA
from torch.nn import Linear
from torch_geometric.utils import negative_sampling
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Set the device (MPS for Mac M1/M2, CPU otherwise)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Define the Encoder and Discriminator classes
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class Discriminator(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = self.lin1(x).relu()
        x = self.lin2(x).relu()
        return self.lin3(x)

# Load dataset
dataset_path = "../pyg_data"  # Set the path to your dataset
dataname = "MUTAG"  # Replace with your dataset name
dataset = TUDataset(os.path.join(dataset_path, dataname), name=dataname)


print(f'Dataset: {dataname}')
print("num of classes: ", dataset.num_classes) #dataset.num_classes

# Split dataset into train and test
def split_dataset(dataset, split_ratio=0.8):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split_point = int(split_ratio * dataset_size)

    # Shuffle indices
    torch.manual_seed(42)  # For reproducibility
    indices = torch.randperm(len(dataset)).tolist()

    # Split indices into training and testing
    train_idx = indices[:split_point]
    test_idx = indices[split_point:]

    # Use indices to create training and testing subsets
    train_dataset = dataset[train_idx]
    test_dataset = dataset[test_idx]

    return train_dataset, test_dataset

train_dataset, test_dataset = split_dataset(dataset)

# Initialize DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize models and move to the device
encoder = Encoder(train_dataset.num_features, hidden_channels=32, out_channels=32).to(device)
discriminator = Discriminator(in_channels=32, hidden_channels=64, out_channels=32).to(device)
model = ARGVA(encoder, discriminator).to(device)

# Optimizers
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.005)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

# Training function
def train():
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)  # Move batch to the device

        # Zero the parameter gradients
        encoder_optimizer.zero_grad()
        discriminator_optimizer.zero_grad()

        # Forward pass through the model
        z = model.encode(batch.x, batch.edge_index)

        # Compute loss
        loss = model.recon_loss(z, batch.edge_index)
        loss += model.kl_loss() / batch.num_graphs  # Add KL divergence loss

        # Backward pass and optimization
        loss.backward()
        encoder_optimizer.step()
        discriminator_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# Corrected testing function
def test():
    model.eval()
    total_auc = 0
    total_ap = 0
    num_batches = 0
    
    for batch in test_loader:
        batch = batch.to(device)
        with torch.no_grad():
            # Encode the data to get latent variables
            z = model.encode(batch.x, batch.edge_index)

            # Generate negative edges for testing
            neg_edge_index = negative_sampling(
                edge_index=batch.edge_index, 
                num_nodes=batch.num_nodes,
                num_neg_samples=batch.edge_index.size(1)
            )

            # Compute AUC and AP scores using positive and negative edges
            auc, ap = model.test(z, batch.edge_index, neg_edge_index)
            
            # Accumulate the AUC and AP scores
            total_auc += auc
            total_ap += ap
            num_batches += 1

    # Calculate the average AUC and AP over all batches
    avg_auc = total_auc / num_batches
    avg_ap = total_ap / num_batches
    
    return avg_auc, avg_ap


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_points(dataset, colors):
    model.eval()

    # List to store graph-level embeddings and corresponding labels
    graph_embeddings = []
    graph_labels = []

    for data in dataset:
        data = data.to(device)
        # Get node-level embeddings using the model's encoder
        z = model.encode(data.x, data.edge_index)

        # Aggregate node embeddings to graph-level embeddings (mean pooling)
        z_graph = z.mean(dim=0).unsqueeze(0)  # Aggregating to graph level

        # Detach the tensor from the computation graph and move to CPU
        z_graph = z_graph.detach().cpu()

        # Append the graph-level embedding and corresponding label
        graph_embeddings.append(z_graph)
        graph_labels.append(data.y.item())

    # Convert the list of graph embeddings to a single tensor and then to NumPy
    graph_embeddings = torch.cat(graph_embeddings, dim=0).numpy()
    graph_labels = torch.tensor(graph_labels).numpy()

    # Apply t-SNE to reduce dimensions to 2 for visualization
    z_2d = TSNE(n_components=2, perplexity=min(30, len(graph_embeddings) - 1)).fit_transform(graph_embeddings)

    # Plotting
    plt.figure(figsize=(8, 8))
    for i in range(dataset.num_classes):
        plt.scatter(z_2d[graph_labels == i, 0], z_2d[graph_labels == i, 1], s=20, color=colors[i], label=f'Class {i}')
    
    plt.axis('off')
    plt.legend()
    plt.show()


# Training loop
for epoch in range(1, 101):
    loss = train()
    avg_auc, avg_ap = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test AUC: {avg_auc:.4f}, Test AP: {avg_ap:.4f}')

# Plot embeddings
colors = [
    '#ffc0cb', '#bada55'
]

plot_points(test_dataset, colors)  # Use the first graph in the test dataset
