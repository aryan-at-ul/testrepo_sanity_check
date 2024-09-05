import os
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, ARGVA
from torch.nn import Linear
from torch_geometric.utils import negative_sampling
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data
from sklearn.cluster import KMeans
from torch_geometric.utils import subgraph
from torch_geometric.nn import global_mean_pool
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

# Ignore certain warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=ConvergenceWarning)


# Set the device
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# GraphComponentModule
class GraphComponentModule(nn.Module):
    def __init__(self, num_clusters):
        super().__init__()
        self.num_clusters = num_clusters

    def forward(self, data):
        # Device handling
        device = data.x.device
        
        x, edge_index = data.x, data.edge_index

        # Clustering nodes using KMeans
        if x.size(0) < self.num_clusters:
            if x.size(0) == 0:
                return []
            new_num_clusters = x.size(0)
        else:
            new_num_clusters = self.num_clusters

        kmeans = KMeans(n_clusters=new_num_clusters)
        labels = kmeans.fit_predict(x.cpu().numpy())
        labels = torch.from_numpy(labels).to(device)

        subgraph_datas = []
        for i in range(new_num_clusters):
            mask = (labels == i).to(device)
            mask_tensor = mask.to(torch.bool)
            sg_nodes = mask_tensor.nonzero(as_tuple=False).squeeze()
            if sg_nodes.dim() == 0:
                sg_nodes = sg_nodes.unsqueeze(0)

            if sg_nodes.numel() > 0:
                sg_edge_index, _ = subgraph(sg_nodes.to(device), edge_index.to(device), relabel_nodes=True, num_nodes=x.size(0))
                sg_data = Data(x=x[sg_nodes], edge_index=sg_edge_index)
                subgraph_datas.append(sg_data)

        return subgraph_datas

# AtomLearningModule
class AtomLearningModule(nn.Module):
    def __init__(self, in_features, hidden_dim, n_layers=5):
        super().__init__()
        self.n_layers = n_layers
        self.conv_layers = nn.ModuleList([GCNConv(in_features if i == 0 else hidden_dim, hidden_dim) for i in range(n_layers)])
        self.pool = global_mean_pool

    def forward(self, subgraph_data):
        x, edge_index, batch = subgraph_data.x, subgraph_data.edge_index, subgraph_data.batch
        
        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = torch.relu(x)

        graph_embedding = self.pool(x, batch)
        return graph_embedding

# Encoder for ARGVA
class GraphMultiComponentEncoder(nn.Module):
    def __init__(self, in_features, hidden_dim, out_dim, num_clusters=5, num_classes=2):
        super().__init__()
        self.num_clusters = num_clusters
        self.component_module = GraphComponentModule(num_clusters=num_clusters)
        self.atom_learner = AtomLearningModule(in_features, hidden_dim)
        self.mu_layer = nn.Linear(hidden_dim * num_clusters, out_dim)
        self.logstd_layer = nn.Linear(hidden_dim * num_clusters, out_dim)
        self.classification_layer = nn.Linear(out_dim, num_classes)
        self.hidden_dim = hidden_dim

    def forward(self, data):
        num_graphs = data.batch.max().item() + 1
        total_embeddings = []

        for graph_idx in range(num_graphs):
            mask = data.batch == graph_idx
            if mask.sum() == 0:
                continue

            graph_data = Data(x=data.x[mask], edge_index=subgraph(mask, data.edge_index)[0], y=data.y[graph_idx])
            subgraphs = self.component_module(graph_data)
            embeddings = [self.atom_learner(sg) for sg in subgraphs]

            if len(embeddings) < self.num_clusters:
                while len(embeddings) < self.num_clusters:
                    embeddings.append(torch.zeros(1, self.hidden_dim, device=graph_data.x.device))

            combined_embedding = torch.cat(embeddings, dim=1)
            total_embeddings.append(combined_embedding)

        batch_embeddings = torch.cat(total_embeddings, dim=0)
        mu = self.mu_layer(batch_embeddings)
        logstd = self.logstd_layer(batch_embeddings)
        return mu, logstd

    def classify(self, z):
        return self.classification_layer(z)

# Discriminator for ARGVA
class Discriminator(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = self.lin1(x).relu()
        x = self.lin2(x).relu()
        return self.lin3(x)

class EnhancedDiscriminator(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout_rate=0.3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers

        # Input Layer
        self.layers.append(nn.Sequential(
            Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout_rate)
        ))

        # Hidden Layers
        for _ in range(1, num_layers - 1):
            self.layers.append(nn.Sequential(
                Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(dropout_rate)
            ))

        # Output Layer
        self.output_layer = Linear(hidden_channels, out_channels)

    def forward(self, x):
        # Forward through all layers
        for layer in self.layers:
            x = layer(x)
        
        # Output layer
        x = self.output_layer(x)
        return x



# Load dataset
dataset_path = "../pyg_data"
dataname = "PROTEINS"
dataset = TUDataset(os.path.join(dataset_path, dataname), name=dataname)

print(f'Dataset: {dataname}')
print("num of classes:", dataset.num_classes)

# Split dataset
def split_dataset(dataset, split_ratio=0.7):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split_point = int(split_ratio * dataset_size)
    torch.manual_seed(42)
    indices = torch.randperm(len(dataset)).tolist()
    train_idx = indices[:split_point]
    test_idx = indices[split_point:]
    train_dataset = dataset[train_idx]
    test_dataset = dataset[test_idx]
    return train_dataset, test_dataset

train_dataset, test_dataset = split_dataset(dataset)

# Initialize DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize models
encoder = GraphMultiComponentEncoder(
    in_features=train_dataset.num_features,
    hidden_dim=64,
    out_dim=32,
    num_clusters=3
).to(device)


# Initialize the enhanced discriminator
enhanced_discriminator = EnhancedDiscriminator(
    in_channels=64,         # Input feature size (matches output of the encoder)
    hidden_channels=64,     # Size of hidden layers
    out_channels=32,        # Output feature size
    num_layers=4,           # Number of layers in the discriminator
    dropout_rate=0.3        # Dropout rate
).to(device)

# Use the custom encoder and enhanced discriminator in the ARGVA model
discriminator = Discriminator(in_channels=64, hidden_channels=64, out_channels=32).to(device)
model = ARGVA(encoder, discriminator).to(device)

# Optimizers
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)


# Training function
def train():
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        encoder_optimizer.zero_grad()
        discriminator_optimizer.zero_grad()

        mu, logstd = model.encoder(batch)
        z = model.reparametrize(mu, logstd)
        logits = model.encoder.classify(z)

        loss = F.cross_entropy(logits, batch.y)
        if torch.isnan(loss).any():
            print("NaN detected in loss. Skipping this batch.")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        encoder_optimizer.step()
        discriminator_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

# Testing function
def test():
    model.eval()
    total_auc = 0
    total_ap = 0
    num_batches = 0
    
    for batch in test_loader:
        batch = batch.to(device)
        with torch.no_grad():
            mu, logstd = model.encoder(batch)
            z = model.reparametrize(mu, logstd)
            logits = model.encoder.classify(z)
            y = batch.y

            y_np = y.detach().cpu().numpy()
            logits_np = logits.detach().cpu().numpy()
            pos_class_scores = logits_np[:, 1]

            if np.isnan(pos_class_scores).any() or np.isnan(y_np).any():
                print("NaN values found in predictions or labels, skipping this batch.")
                continue

            auc = roc_auc_score(y_np, pos_class_scores)
            ap = average_precision_score(y_np, pos_class_scores)

            total_auc += auc
            total_ap += ap
            num_batches += 1

    avg_auc = total_auc / num_batches if num_batches > 0 else 0
    avg_ap = total_ap / num_batches if num_batches > 0 else 0
    
    return avg_auc, avg_ap

# Plotting function
def plot_points(loader, colors):
    model.eval()
    graph_embeddings = []
    graph_labels = []

    for batch in loader:
        batch = batch.to(device)
        mu, logstd = model.encoder(batch)
        z = model.reparametrize(mu, logstd)

        if z.size(0) != batch.y.size(0):
            raise ValueError(f"Size mismatch: z has {z.size(0)} graph embeddings, but batch.y has {batch.y.size(0)} labels.")

        z = z.detach().cpu()
        graph_embeddings.append(z)
        graph_labels.extend(batch.y.cpu().numpy())

    graph_embeddings = torch.cat(graph_embeddings, dim=0).numpy()
    graph_labels = np.array(graph_labels)

    z_2d = TSNE(n_components=2, perplexity=min(30, len(graph_embeddings) - 1)).fit_transform(graph_embeddings)

    plt.figure(figsize=(8, 8))
    for i in range(len(colors)):
        plt.scatter(z_2d[graph_labels == i, 0], z_2d[graph_labels == i, 1], s=20, color=colors[i], label=f'Class {i}')
    
    plt.axis('off')
    plt.legend()
    plt.show()

# Training loop
for epoch in range(1, 101):
    loss = train()
    avg_auc, avg_ap = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test AUC: {avg_auc:.4f}, Test AP: {avg_ap:.4f}')

colors = ['#ff0000', '#0000ff']
plot_points(test_loader, colors)
