import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

# def calculate_random_walk_matrix(adj):
#     """
#     Returns the random walk adjacency matrix. This is for D_GCN
#     """
#     # 确保adj_mx是一个稀疏张量
#     if not adj.is_sparse:
#         adj = adj.to_sparse()
#     # 计算度矩阵D
#     d = torch.sparse.sum(adj, dim=1).to_dense()
#     d = torch.where(d == 0, torch.ones_like(d), d)

#     # 计算D的逆
#     d_inv = torch.pow(d, -1).flatten()
#     d_inv[torch.isinf(d_inv)] = 0.

#     # 构造对角矩阵D^-1
#     d_mat_inv = torch.diag(d_inv).to(adj.device)

#     # 计算随机游走矩阵D^-1 * A
#     random_walk_mx = torch.mm(d_mat_inv, adj.to_dense())

#     return random_walk_mx


def calculate_random_walk_matrix(adj):
    """
    Returns the random walk adjacency matrix. This is for D_GCN
    """
    device = adj.device
    adj = adj.cpu().numpy()
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d = np.where(d == 0, 1, d)
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj).tocoo()  #type:ignore
    return torch.from_numpy(random_walk_mx.toarray()).float().to(device)


class D_GCN(nn.Module):
    """
    Neural network block that applies a diffusion graph convolution to sampled location
    """

    def __init__(self, in_channels, out_channels, k_hop, activation='relu'):
        """
        :param in_channels: Number of time step.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param order: The diffusion steps.
        """
        super(D_GCN, self).__init__()
        self.orders = k_hop
        if activation != 'identity':
            self.activation = getattr(F, activation)
        else:
            self.activation = nn.Identity()
        self.num_matrices = 2 * self.orders + 1
        self.Theta1 = nn.Parameter(
            torch.FloatTensor(in_channels * self.num_matrices, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)
        stdv1 = 1. / math.sqrt(self.bias.shape[0])
        self.bias.data.uniform_(-stdv1, stdv1)

    def _concat(self, x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def forward(self, X, A_q, A_h):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps)
        :A_q: The forward random walk matrix (num_nodes, num_nodes)
        :A_h: The backward random walk matrix (num_nodes, num_nodes)
        :return: Output data of shape (batch_size, num_nodes, num_features)
        """
        batch_size = X.shape[0]  # batch_size
        num_node = X.shape[1]
        input_size = X.size(2)  # time_length
        supports = []
        supports.append(A_q)
        supports.append(A_h)

        x0 = X.permute(1, 2, 0)  #(num_nodes, num_times, batch_size)
        x0 = torch.reshape(x0, shape=[num_node, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)
        for support in supports:
            x1 = torch.mm(support, x0)
            x = self._concat(x, x1)
            for k in range(2, self.orders + 1):
                x2 = 2 * torch.mm(support, x1) - x0
                x = self._concat(x, x2)
                x1, x0 = x2, x1

        x = torch.reshape(
            x, shape=[self.num_matrices, num_node, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(
            x, shape=[batch_size, num_node, input_size * self.num_matrices])
        x = torch.matmul(
            x, self.Theta1)  # (batch_size * self._num_nodes, output_size)
        x += self.bias
        x = self.activation(x)
        return x


class C_GCN(nn.Module):
    """
    Neural network block that applies a chebynet to sampled location.
    """

    def __init__(self, in_channels, out_channels, orders, activation='relu'):
        """
        :param in_channels: Number of time step.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param order: The order of convolution
        :param num_nodes: Number of nodes in the graph.
        """
        super(C_GCN, self).__init__()
        self.Theta1 = nn.Parameter(
            torch.FloatTensor(in_channels * orders, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.orders = orders
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)
        stdv1 = 1. / math.sqrt(self.bias.shape[0])
        self.bias.data.uniform_(-stdv1, stdv1)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps)
        :A_hat: The Laplacian matrix (num_nodes, num_nodes)
        :return: Output data of shape (batch_size, num_nodes, num_features)
        """
        list_cheb = list()
        for k in range(self.orders):
            if (k == 0):
                list_cheb.append(torch.diag(torch.ones(A_hat.shape[0], )))
            elif (k == 1):
                list_cheb.append(A_hat)
            else:
                list_cheb.append(2 * torch.matmul(A_hat, list_cheb[k - 1]) -
                                 list_cheb[k - 2])

        features = list()
        for k in range(self.orders):
            features.append(torch.einsum("kk,bkj->bkj", [list_cheb[k], X]))
        features_cat = torch.cat(features, 2)
        t2 = torch.einsum("bkj,jh->bkh", [features_cat, self.Theta1])
        t2 += self.bias
        if self.activation == 'relu':
            t2 = F.relu(t2)
        if self.activation == 'selu':
            t2 = F.selu(t2)
        return t2


class GCNLayer(nn.Module):

    def __init__(self, in_features, out_features, use_bias=True):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

        self.weight = nn.Parameter(torch.FloatTensor(in_features,
                                                     out_features))
        if self.use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, X, adj):
        """
        Args:
            X (torch.Tensor): 节点特征矩阵，维度为 [num_nodes, in_features]
            adj (torch.Tensor): 带权邻接矩阵，维度为 [num_nodes, num_nodes]

        Returns:
            torch.Tensor: 更新后的节点特征矩阵，维度为 [num_nodes, out_features]
        """
        # 归一化邻接矩阵
        adj = (adj + adj.T) / 2
        D = torch.diag(torch.pow(adj.sum(1), -0.5))
        adj_normalized = D @ adj @ D

        # GCN层计算
        support = torch.mm(X, self.weight)
        output = torch.mm(adj_normalized, support)

        if self.bias is not None:
            output += self.bias

        return output


# 定义简单的两层GCN模型
class GCN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, output_dim)

    def forward(self, x, adj_mask, adj):
        x = self.gcn1(x, adj_mask)
        x = F.relu(x)
        x = self.gcn2(x, adj)
        return x


class K_GCN(nn.Module):
    """
    Neural network block that applies a graph convolution to to sampled location.
    """

    def __init__(self, in_channels, out_channels, activation='selu'):
        """
        :param in_channels: Number of time step.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        :relu is not good for K_GCN on Kriging, so we suggest 'selu' 
        """
        super(K_GCN, self).__init__()
        self.Theta1 = nn.Parameter(torch.FloatTensor(in_channels,
                                                     out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)
        stdv1 = 1. / math.sqrt(self.bias.shape[0])
        self.bias.data.uniform_(-stdv1, stdv1)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps)
        :A_hat: The normalized adajacent matrix (num_nodes, num_nodes)
        :return: Output data of shape (batch_size, num_nodes, num_features)
        """
        features = torch.einsum("kk,bkj->bkj", [A_hat, X])
        t2 = torch.einsum("bkj,jh->bkh", [features, self.Theta1])
        t2 += self.bias
        if self.activation == 'relu':
            t2 = F.relu(t2)
        if self.activation == 'selu':
            t2 = F.selu(t2)

        return t2


class GATLayer(nn.Module):
    """
    Neural network block that applies attention mechanism to sampled locations (only the attention).
    """

    def __init__(self, in_channels, alpha, threshold, activate=True):
        """
        :param in_channels: Number of time step.
        :param alpha: alpha for leaky Relu.
        :param threshold: threshold for graph connection
        :param concat: whether concat features
        :It should be noted that the input layer should use linear activation
        """
        super(GATLayer, self).__init__()
        self.alpha = alpha
        self.threshold = threshold
        self.in_channels = in_channels
        self.W = nn.Parameter(torch.zeros(size=(in_channels, in_channels)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * in_channels, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.activate = activate
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):

        # num of nodes
        h = torch.matmul(input, self.W)
        B = h.size()[0]
        N = h.size()[1]

        a_input = torch.cat([
            h.repeat(1, 1, N).view(B, N * N, self.in_channels),
            h.repeat(1, N, 1)
        ],
                            dim=2).view(B, N, N, 2 * self.in_channels)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        zero_vec = -9e15 * torch.ones_like(e)

        attention = torch.where(
            adj.unsqueeze(0).repeat(B, 1, 1) > self.threshold, e,
            zero_vec)  #>threshold for attention connection
        attention = F.softmax(attention, dim=2)
        h_prime = torch.matmul(attention, h)
        if self.activate:
            return F.elu(h_prime)
        else:
            return h_prime


class GATv2Layer(nn.Module):

    def __init__(self, in_features, out_features, alpha, activate=True):
        super(GATv2Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.activate = activate
        self.linear = nn.Linear(in_features, out_features)

        # 定义可学习的权重矩阵 W
        self.W = nn.Linear(2 * out_features, out_features, bias=False)

        # 定义注意力机制的参数向量 a
        self.a = nn.Parameter(torch.empty(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        B, N, _ = h.size()
        h = self.linear(h)

        # 创建注意力机制的输入
        a_input = torch.cat(
            [h.repeat(1, 1, N).view(B, N * N, -1),
             h.repeat(1, N, 1)], dim=2).view(B, N, N, 2 * self.out_features)

        # 对拼接特征进行线性变换并应用LeakyReLU
        Wh_ij = self.leakyrelu(self.W(a_input))

        # 计算注意力系数
        e = torch.matmul(Wh_ij, self.a).squeeze(3)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)

        h_prime = torch.matmul(attention, h)

        if self.activate:
            return F.elu(h_prime)
        else:
            return h_prime
