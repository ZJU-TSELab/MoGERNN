import torch
import torch.nn as nn
from layers.gnn import D_GCN, calculate_random_walk_matrix, GCNLayer


class MeanAggregator(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(MeanAggregator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

    def forward(self, adj, x, batch_unknown_nodes, *args):
        adj = adj.clone()
        x = self.fc1(x)
        adj[:, batch_unknown_nodes] = 0
        unweighted_adj = torch.where(adj > 0, 1., 0.)
        # Degree matrix (sum of each row of adj gives the degree of each node)
        degree = torch.sum(unweighted_adj, dim=1, keepdim=True)  # [N, 1]
        # Avoid division by zero
        degree[degree == 0] = 1
        # Normalize adjacency matrix (mean aggregation)
        norm_adj = adj / degree  # [N, N]
        # Aggregate neighbors' features
        aggregated_features = torch.matmul(norm_adj, x)  # [B, N, input_dim]
        # Apply linear transformation
        output = self.fc2(aggregated_features)  # [B, N, hidden_dim]

        return output


class WMeanAggregator(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(WMeanAggregator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

    def forward(self, adj, x, batch_unknown_nodes, *args):
        adj = adj.clone()
        x = self.fc1(x)
        adj[:, batch_unknown_nodes] = 0
        # Degree matrix (sum of each row of adj gives the degree of each node)
        degree = torch.sum(adj, dim=1, keepdim=True)  # [N, 1]
        # Avoid division by zero
        degree[degree == 0] = 1
        # Normalize adjacency matrix (mean aggregation)
        norm_adj = adj / degree  # [N, N]
        # Aggregate neighbors' features
        aggregated_features = torch.matmul(norm_adj, x)  # [B, N, input_dim]
        # Apply linear transformation
        output = self.fc2(aggregated_features)  # [B, N, hidden_dim]

        return output


class MaxPoolingAggregator(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(MaxPoolingAggregator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

    def forward(self, adj, x, batch_unknown_nodes, *args):
        adj = adj.clone()
        x = self.fc1(x)
        adj[:, batch_unknown_nodes] = 0
        adj = torch.where(adj > 0, 1., 0.)
        x = x.permute(0, 2, 1)
        neighbor_features = torch.einsum('ijk,kl->ijkl', x,
                                         adj).permute(0, 1, 3, 2)
        max_pooled_features = torch.max(neighbor_features, dim=-1).values
        max_pooled_features = max_pooled_features.permute(0, 2, 1)
        out_features = self.fc2(max_pooled_features)
        return out_features


class MinPoolingAggregator(nn.Module):
    """minpooling的adj应该比maxpolling的adj范围小?因为拥堵波在有限时间里传的更慢"""

    def __init__(self, input_dim, hidden_dim):
        super(MinPoolingAggregator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

    def forward(self, adj, x, batch_unknown_nodes, *args):
        adj = adj.clone()
        x = self.fc1(x)
        adj[:, batch_unknown_nodes] = 0
        adj = torch.where(adj > 0, 1., 0.)
        x = x.permute(0, 2, 1)
        neighbor_features = torch.einsum('ijk,kl->ijkl', x,
                                         adj).permute(0, 1, 3, 2)
        max_pooled_features = torch.min(neighbor_features, dim=-1).values
        max_pooled_features = max_pooled_features.permute(0, 2, 1)
        out_features = self.fc2(max_pooled_features)
        return out_features


class GCNAggregator(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(GCNAggregator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.conv = GCNLayer(hidden_dim, hidden_dim)
        self.fc2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

    def forward(self, adj, x, batch_unknown_nodes, *args):
        adj = adj.clone()
        adj[:, batch_unknown_nodes] = 0
        x = self.fc1(x)
        x = self.conv(x, adj)
        x = self.fc2(x)
        return x


class DiffusionAggregator(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(DiffusionAggregator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.conv = D_GCN(hidden_dim, hidden_dim, 1, activation='identity')
        self.fc2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

    def forward(self, adj, x, batch_unknown_nodes, *args):
        if len(args) == 0:
            adj = adj.clone()
            adj_h = adj.T.clone()
            adj[:, batch_unknown_nodes] = 0
            adj_h[:, batch_unknown_nodes] = 0
            A_q = calculate_random_walk_matrix(adj)
            A_h = calculate_random_walk_matrix(adj_h)
        else:
            A_q, A_h = args
        x = self.fc1(x)
        x = self.conv(x, A_q, A_h)
        x = self.fc2(x)
        return x
