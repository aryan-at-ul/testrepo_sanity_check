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



import warnings
warnings.filterwarnings("ignore", message="The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning")




# class GraphComponentModule(nn.Module):
#     def __init__(self, num_clusters):
#         super().__init__()
#         self.num_clusters = num_clusters

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         # print("modify input data to subgraphs",x.shape)

#         # Check if the number of samples is less than the number of clusters
#         if x.size(0) < self.num_clusters:
#             if x.size(0) == 0:
#                 return []  # Return an empty list if there are no nodes
#             new_num_clusters = x.size(0)  # Reduce clusters to match available nodes
#         else:
#             new_num_clusters = self.num_clusters

#         # Applying k-means clustering to node features
#         kmeans = KMeans(n_clusters=new_num_clusters)
#         labels = kmeans.fit_predict(x.cpu().numpy())  # Ensure x is on CPU and can be converted to NumPy array

#         subgraph_datas = []
#         for i in range(new_num_clusters):
#             mask = labels == i
#             mask_tensor = torch.from_numpy(mask).to(torch.bool)  # Convert numpy boolean array to a PyTorch tensor
#             sg_nodes = mask_tensor.nonzero().squeeze()  # Get the indices of nonzero elements
#             if sg_nodes.dim() == 0:
#                 sg_nodes = sg_nodes.unsqueeze(0)  # Add a batch dimension if needed

#             if sg_nodes.numel() > 0:
#                 sg_edge_index, sg_edge_attr = subgraph(sg_nodes, edge_index, relabel_nodes=True, num_nodes=x.size(0))
#                 sg_data = Data(x=x[sg_nodes], edge_index=sg_edge_index)
#                 subgraph_datas.append(sg_data)

#         return subgraph_datas


# class AtomLearningModule(nn.Module):
#     def __init__(self, in_features, hidden_dim, n_layers=5):
#         super().__init__()
#         self.n_layers = n_layers
        

#         self.conv_layers = nn.ModuleList([GCNConv(in_features if i == 0 else hidden_dim, hidden_dim) for i in range(n_layers)])
#         self.pool = global_add_pool

#     def forward(self, subgraph_data):
#         x, edge_index, batch = subgraph_data.x, subgraph_data.edge_index, subgraph_data.batch
        
#         # Apply each GCN layer and accumulate graph embeddings
#         for conv in self.conv_layers:
#             x = conv(x, edge_index)
#             x = torch.relu(x) 


#         graph_embedding = self.pool(x, batch)
#         return graph_embedding




# class GraphMultiComponentClassifier(nn.Module):
#     def __init__(self, in_features, hidden_dim, num_classes, num_clusters=5):
#         super().__init__()
#         self.num_clusters = num_clusters
#         self.component_module = GraphComponentModule(num_clusters=num_clusters)
#         self.atom_learner = AtomLearningModule(in_features, hidden_dim)
#         self.classifier = nn.Linear(hidden_dim * num_clusters, num_classes)

#     def forward(self, data):
#         batch_size = data.num_graphs
#         # print("batch size",batch_size)
#         total_embeddings = []

#         for b in range(batch_size):
#             subgraphs = self.component_module(data[b])
#             embeddings = [self.atom_learner(sg) for sg in subgraphs]

#             # Check if we have fewer embeddings than expected
#             if len(embeddings) < self.num_clusters:
#                 # Pad the embeddings list with zeros tensors to reach the expected number
#                 while len(embeddings) < self.num_clusters:
#                     embeddings.append(torch.zeros(1, 64, device=data.x.device))

#             combined_embedding = torch.cat(embeddings, dim=1)  # Concatenate along feature dimension
#             total_embeddings.append(combined_embedding)

#         batch_embeddings = torch.cat(total_embeddings, dim=0)
#         # print("batch embeddings",batch_embeddings.shape)
#         out = self.classifier(batch_embeddings)

#         return F.log_softmax(out, dim=-1)



class GraphComponentModule(nn.Module):
    def __init__(self, num_clusters):
        super().__init__()
        self.num_clusters = num_clusters

    def forward(self, data):
        # Get the device from the input data tensor
        device = data.x.device  # Assuming data.x is a PyTorch tensor already on the correct device
        
        x, edge_index = data.x, data.edge_index

        # Check if the number of samples is less than the number of clusters
        if x.size(0) < self.num_clusters:
            if x.size(0) == 0:
                return []  # Return an empty list if there are no nodes
            new_num_clusters = x.size(0)  # Reduce clusters to match available nodes
        else:
            new_num_clusters = self.num_clusters

        # Applying k-means clustering to node features (KMeans expects CPU tensors)
        kmeans = KMeans(n_clusters=new_num_clusters)
        labels = kmeans.fit_predict(x.cpu().numpy())  # Ensure x is on CPU and can be converted to NumPy array

        # Convert labels back to a tensor on the correct device
        labels = torch.from_numpy(labels).to(device)

        subgraph_datas = []
        for i in range(new_num_clusters):
            mask = (labels == i).to(device)  # Ensure mask is on the correct device
            mask_tensor = mask.to(torch.bool)  # Convert to boolean tensor if needed
            sg_nodes = mask_tensor.nonzero(as_tuple=False).squeeze()  # Get the indices of nonzero elements
            if sg_nodes.dim() == 0:
                sg_nodes = sg_nodes.unsqueeze(0)  # Add a batch dimension if needed

            if sg_nodes.numel() > 0:
                # Ensure edge_index is on the correct device
                sg_edge_index, sg_edge_attr = subgraph(
                    sg_nodes.to(device), edge_index.to(device), relabel_nodes=True, num_nodes=x.size(0)
                )
                sg_data = Data(x=x[sg_nodes], edge_index=sg_edge_index)
                subgraph_datas.append(sg_data)

        return subgraph_datas



class AtomLearningModule(nn.Module):
    def __init__(self, in_features, hidden_dim, n_layers=7):
        super().__init__()
        self.n_layers = n_layers
        self.conv_layers = nn.ModuleList([GCNConv(in_features if i == 0 else hidden_dim, hidden_dim) for i in range(n_layers)])
        self.pool = global_mean_pool
        self.margin = 1.0  # Margin for the contrastive loss

    def forward(self, subgraph_data):
        x, edge_index, batch = subgraph_data.x, subgraph_data.edge_index, subgraph_data.batch
        
        # Apply each GCN layer and accumulate graph embeddings
        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = torch.relu(x) 

        graph_embedding = self.pool(x, batch)
        return graph_embedding

    def compute_contrastive_loss(self, embeddings):
        """
        Compute a contrastive loss to maximize the distance between subgraph embeddings.
        """
        num_embeddings = len(embeddings)
        if num_embeddings < 2:
            return 0.0  # No contrastive loss if there is less than two embeddings

        loss = 0.0
        for i in range(num_embeddings):
            for j in range(i + 1, num_embeddings):
                dist = torch.norm(embeddings[i] - embeddings[j], p=2)  # Euclidean distance
                loss += F.relu(self.margin - dist)  # Contrastive loss encourages embeddings to be farther apart

        return loss / (num_embeddings * (num_embeddings - 1) / 2)  # Normalize the loss by the number of pairs

    def compute_orthogonality_loss(self, embeddings):
        """
        Compute a pair-wise orthogonality loss to maximize the orthogonality between subgraph embeddings.
        """
        # Convert embeddings list to a PyTorch tensor
        if isinstance(embeddings, list):
            embeddings = torch.stack(embeddings)

        num_embeddings = embeddings.size(0)
        if num_embeddings < 2:
            return 0.0  # No orthogonality loss if there are fewer than two embeddings

        # Normalize embeddings to unit vectors along the last dimension
        normalized_embeddings = F.normalize(embeddings, p=2, dim=-1)

        # Compute pairwise dot products using batch matrix multiplication
        dot_product_matrix = torch.bmm(normalized_embeddings, normalized_embeddings.transpose(1, 2))

        # Compute orthogonality loss by summing the squared off-diagonal elements
        # Subtract the identity matrix to remove diagonal elements (self-dot products)
        identity_matrix = torch.eye(dot_product_matrix.size(1), device=dot_product_matrix.device).unsqueeze(0)
        orthogonality_loss = torch.sum((dot_product_matrix - identity_matrix) ** 2)

        # Normalize the loss by the number of off-diagonal elements
        return orthogonality_loss / (num_embeddings * (num_embeddings - 1))

    def compute_combined_loss(self, embeddings, alpha=0.5):
        """
        Compute a combined loss of contrastive and orthogonality to maximize diversity and distance.
        """
        contrastive_loss = self.compute_contrastive_loss(embeddings)
        orthogonality_loss = self.compute_orthogonality_loss(embeddings)

        # Combine losses with a weighting factor alpha
        combined_loss = alpha * contrastive_loss + (1 - alpha) * orthogonality_loss

        return combined_loss




class GraphMultiComponentClassifier(nn.Module):
    def __init__(self, in_features, hidden_dim, num_classes, num_clusters=5):
        super().__init__()
        self.num_clusters = num_clusters
        self.component_module = GraphComponentModule(num_clusters=num_clusters)
        self.atom_learner = AtomLearningModule(in_features, hidden_dim)
        self.classifier = nn.Linear(hidden_dim * num_clusters, num_classes)

    def forward(self, data):
        batch_size = data.num_graphs
        total_embeddings = []
        contrastive_loss = 0.0

        for b in range(batch_size):
            subgraphs = self.component_module(data[b])
            embeddings = [self.atom_learner(sg) for sg in subgraphs]

            # Check if we have fewer embeddings than expected
            if len(embeddings) < self.num_clusters:
                # Pad the embeddings list with zeros tensors to reach the expected number
                while len(embeddings) < self.num_clusters:
                    embeddings.append(torch.zeros(1, 64, device=data.x.device))

            # Compute contrastive loss
            # contrastive_loss += self.atom_learner.compute_contrastive_loss(embeddings)
            # contrastive_loss += self.atom_learner.compute_combined_loss(embeddings)
            contrastive_loss += self.atom_learner.compute_orthogonality_loss(embeddings)


            combined_embedding = torch.cat(embeddings, dim=1)  # Concatenate along feature dimension
            total_embeddings.append(combined_embedding)

        batch_embeddings = torch.cat(total_embeddings, dim=0)
        out = self.classifier(batch_embeddings)

        # Return both the classification output and the contrastive loss
        return F.log_softmax(out, dim=-1), contrastive_loss


