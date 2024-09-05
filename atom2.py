import torch
from sklearn.cluster import KMeans
from torch_geometric.utils import subgraph
from torch_geometric.data import Data
import os
import torch
from torch_geometric.data import DataLoader, InMemoryDataset, Data
import torch_geometric.transforms as GT
from torch_geometric.nn import knn_graph
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

import torch_geometric.transforms as GT
from torch_geometric.utils import to_networkx
import networkx as nx
import numpy as np
import numpy as np
import matplotlib.pylab as pl
from sklearn.manifold import MDS
# from ot.gromov import gromov_wasserstein_linear_unmixing, gromov_wasserstein_dictionary_learning, fused_gromov_wasserstein_linear_unmixing, fused_gromov_wasserstein_dictionary_learning
# import ot
import networkx
from networkx.generators.community import stochastic_block_model as sbm
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch_geometric.data import Data
from sklearn.cluster import KMeans
from torch_geometric.utils import to_undirected, subgraph
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool

import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress only the specific ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.cluster import KMeans



class GraphComponentModule(nn.Module):
    def __init__(self, num_clusters):
        super().__init__()
        self.num_clusters = num_clusters

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Check if the number of samples is less than the number of clusters
        if x.size(0) < self.num_clusters:
            if x.size(0) == 0:
                return []  # Return an empty list if there are no nodes
            new_num_clusters = x.size(0)  # Reduce clusters to match available nodes
        else:
            new_num_clusters = self.num_clusters

        # Applying k-means clustering to node features
        kmeans = KMeans(n_clusters=new_num_clusters)
        labels = kmeans.fit_predict(x.cpu().numpy())  # Ensure x is on CPU and can be converted to NumPy array

        subgraph_datas = []
        for i in range(new_num_clusters):
            mask = labels == i
            mask_tensor = torch.from_numpy(mask).to(torch.bool)  # Convert numpy boolean array to a PyTorch tensor
            sg_nodes = mask_tensor.nonzero().squeeze()  # Get the indices of nonzero elements
            if sg_nodes.dim() == 0:
                sg_nodes = sg_nodes.unsqueeze(0)  # Add a batch dimension if needed

            if sg_nodes.numel() > 0:
                sg_edge_index, _ = subgraph(sg_nodes, edge_index, relabel_nodes=True, num_nodes=x.size(0))
                sg_data = Data(x=x[sg_nodes], edge_index=sg_edge_index)
                subgraph_datas.append(sg_data)

        return subgraph_datas


class AtomLearningModule(nn.Module):
    def __init__(self, in_features, hidden_dim, n_layers=5):
        super().__init__()
        self.n_layers = n_layers
        self.conv_layers = nn.ModuleList([GCNConv(in_features if i == 0 else hidden_dim, hidden_dim) for i in range(n_layers)])
        self.pool = global_add_pool

    def forward(self, subgraph_data):
        x, edge_index, batch = subgraph_data.x, subgraph_data.edge_index, subgraph_data.batch
        
        # Apply each GCN layer and accumulate graph embeddings
        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = torch.relu(x)

        # Aggregate the node embeddings into a subgraph embedding
        subgraph_embedding = self.pool(x, batch)
        return subgraph_embedding



class DictionaryLearningModule(nn.Module):
    def __init__(self, feature_dim, num_atoms, device='cpu'):
        super().__init__()
        # Initialize the dictionary (atoms) with the correct shape: (feature_dim, num_atoms)
        self.dictionary = nn.Parameter(torch.randn(feature_dim, num_atoms, device=device))
        self.num_atoms = num_atoms
        self.feature_dim = feature_dim
        self.device = device

    def forward(self, subgraph_embeddings):
        """
        Generate coefficients (sparse codes) to represent subgraph embeddings using learned dictionary atoms.
        Args:
            subgraph_embeddings (Tensor): Subgraph embeddings [num_subgraphs, 1, feature_dim]

        Returns:
            sparse_codes (Tensor): Sparse codes representing the subgraph embeddings [num_subgraphs, num_atoms]
        """
        num_subgraphs = subgraph_embeddings.size(0)  # Should be 4
        sparse_codes = torch.zeros((num_subgraphs, self.num_atoms), device=self.device)

        # The dictionary should be of shape (feature_dim, num_atoms) for the least squares
        # It does not need to be transposed, keep it as (64, 10)
        dictionary = self.dictionary  # Shape should be (64, 10)

        # For each subgraph, compute the coefficients for the best representation using the dictionary atoms
        for i in range(num_subgraphs):
            y_sub = subgraph_embeddings[i].squeeze(0)  # Convert from shape [1, 64] to [64]

            # Solve for sparse coefficients using lstsq (least squares)
            lstsq_result = torch.linalg.lstsq(dictionary, y_sub)  # Solves for x in Ax = y where A=dictionary
            coeff = lstsq_result.solution

            sparse_codes[i] = coeff

        return sparse_codes


class GraphMultiComponentClassifier(nn.Module):
    def __init__(self, in_features, hidden_dim, num_classes, num_clusters=5, num_atoms=30):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_clusters = num_clusters
        self.component_module = GraphComponentModule(num_clusters=num_clusters)
        self.atom_learner = AtomLearningModule(in_features, hidden_dim)
        self.dict_module = DictionaryLearningModule(hidden_dim, num_atoms)
        self.classifier = nn.Linear(num_clusters * (2 * hidden_dim), num_classes)

    def forward(self, data):
        batch_size = data.num_graphs  # Number of graphs in the batch
        all_graph_embeddings = []

        for b in range(batch_size):
            subgraphs = self.component_module(data[b])  # Extract subgraphs from the current graph
            subgraph_features = []

            # Process each subgraph to get embeddings
            for sg in subgraphs:
                sg_embedding = self.atom_learner(sg)  # Learn the embedding for each subgraph
                subgraph_features.append(sg_embedding)

            # Stack subgraph embeddings into a tensor
            subgraph_features_tensor = torch.stack(subgraph_features).squeeze(1)  # Shape: [num_subgraphs, hidden_dim]

            # Compute sparse codes using dictionary learning
            sparse_codes = self.dict_module(subgraph_features_tensor)  # Shape: [num_subgraphs, num_atoms]

            # Reweight each subgraph's features using the atoms
            atoms = self.dict_module.dictionary  # Shape: [hidden_dim, num_atoms]
            reweighted_features = torch.mm(sparse_codes, atoms.T)  # Shape: [num_subgraphs, hidden_dim]

            # Combine reweighted features with the original subgraph features
            combined_features = torch.cat((subgraph_features_tensor, reweighted_features), dim=1)  # Shape: [num_subgraphs, 2 * hidden_dim]

            # Adjust to ensure we have exactly `num_clusters` subgraphs for each graph
            if combined_features.size(0) < self.num_clusters:
                # If fewer subgraphs, pad with zeros
                padding = torch.zeros((self.num_clusters - combined_features.size(0), combined_features.size(1)), device=combined_features.device)
                combined_features = torch.cat([combined_features, padding], dim=0)
            elif combined_features.size(0) > self.num_clusters:
                # If more subgraphs, truncate to match `num_clusters`
                combined_features = combined_features[:self.num_clusters, :]

            all_graph_embeddings.append(combined_features)

        # Stack all graph embeddings to form the batch
        batch_embeddings = torch.stack(all_graph_embeddings)  # Shape: [batch_size, num_clusters, 2 * hidden_dim]

        # Flatten the batch embeddings for classification
        batch_embeddings = batch_embeddings.view(batch_size, -1)  # Reshape to [batch_size, num_clusters * (2 * hidden_dim)]

        # Ensure the input size matches the classifier's expected size
        expected_input_size = self.num_clusters * (2 * self.hidden_dim)
        if batch_embeddings.size(1) != expected_input_size:
            raise ValueError(f"Expected input size for the classifier: {expected_input_size}, but got: {batch_embeddings.size(1)}")

        # Classify using the learned dictionary representation
        logits = self.classifier(batch_embeddings)

        return F.log_softmax(logits, dim=-1),0

