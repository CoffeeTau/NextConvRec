# item_graph_embedder.py
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from collections import defaultdict

class GCN(torch.nn.Module):
    def __init__(self, num_nodes, hidden_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_nodes, hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, edge_index):
        x = self.embedding.weight
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x  # shape: [num_items, hidden_dim]

def build_item_graph(user_seqs, num_items):
    edge_set = set()
    for seq in user_seqs:
        for i in range(len(seq) - 1):
            edge_set.add((seq[i], seq[i+1]))
            edge_set.add((seq[i+1], seq[i]))  # optional: undirected

    src_nodes = [e[0] for e in edge_set]
    tgt_nodes = [e[1] for e in edge_set]
    edge_index = torch.tensor([src_nodes, tgt_nodes], dtype=torch.long)
    return edge_index
