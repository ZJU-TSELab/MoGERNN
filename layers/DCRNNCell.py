import math
import torch
from torch import nn
import torch.nn.functional as F
from .GRUCell import MoGEGRUCellBase, GraphGRUCellBase
from layers.agg import MeanAggregator, WMeanAggregator, MaxPoolingAggregator, MinPoolingAggregator, DiffusionAggregator


class DiffConv(nn.Module):
    """
    Neural network block that applies a diffusion graph convolution to sampled location
    """

    def __init__(self, in_channels, out_channels, k_hop):
        """
        :param in_channels: Number of time step.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param order: The diffusion steps.
        """
        super(DiffConv, self).__init__()
        self.orders = k_hop
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
        x = x + self.bias
        return x


class DCRNNCell(GraphGRUCellBase):
    """The Diffusion Convolutional Recurrent cell from the paper
    `"Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic
    Forecasting" <https://arxiv.org/abs/1707.01926>`_ (Li et al., ICLR 2018).

    Args:
        input_size: Size of the input.
        hidden_size: Number of units in the hidden state.
        k: Size of the diffusion kernel.
        root_weight: Whether to learn a separate transformation for the central
            node.
    """

    def __init__(self, input_size: int, hidden_size: int, k: int = 2):
        # instantiate gates
        forget_gate = DiffConv(input_size + hidden_size, hidden_size, k)
        update_gate = DiffConv(input_size + hidden_size, hidden_size, k)
        candidate_gate = DiffConv(input_size + hidden_size, hidden_size, k)
        super(DCRNNCell, self).__init__(hidden_size=hidden_size,
                                        forget_gate=forget_gate,
                                        update_gate=update_gate,
                                        candidate_gate=candidate_gate)


class MoGERNNCell(MoGEGRUCellBase):

    def __init__(self, experts_list, input_dim, hidden_dim, num_used_experts):
        forget_gate = MoGE(experts_list, input_dim + hidden_dim, hidden_dim,
                           num_used_experts)
        update_gate = MoGE(experts_list, input_dim + hidden_dim, hidden_dim,
                           num_used_experts)
        candidate_gate = MoGE(
            experts_list,
            input_dim + hidden_dim,
            hidden_dim,
            num_used_experts,
        )
        super(MoGERNNCell, self).__init__(hidden_size=hidden_dim,
                                          forget_gate=forget_gate,
                                          update_gate=update_gate,
                                          candidate_gate=candidate_gate)


class SparseGatingNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_experts, num_used_experts):
        super(SparseGatingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_experts)
        self.k = num_used_experts

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入特征，维度为 [batch_size, input_dim]

        Returns:
            torch.Tensor: 每个专家的权重，维度为 [batch_size, num_experts]
        """
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        topk_values, topk_indices = torch.topk(logits, self.k, dim=-1)
        mask = torch.zeros_like(logits).scatter_(-1, topk_indices, 1.0)
        sparse_logits = logits * mask
        weights = F.softmax(sparse_logits, dim=-1)
        return weights


class GraphExpert(nn.Module):

    def __init__(self, gnn: nn.Module):
        super(GraphExpert, self).__init__()
        self.gnn = gnn

    def forward(self, x_enc, adj, batch_unknown_nodes, *args):
        return self.gnn(adj, x_enc, batch_unknown_nodes, *args)


class MoGE(nn.Module):

    def __init__(self, experts_list, input_dim, hidden_dim, num_used_experts):
        super(MoGE, self).__init__()
        experts = []
        if 'mean_expert' in experts_list:
            experts.append(MeanAggregator(input_dim, hidden_dim))
        if 'weight_expert' in experts_list:
            experts.append(WMeanAggregator(input_dim, hidden_dim))
        if 'max_expert' in experts_list:
            experts.append(MaxPoolingAggregator(input_dim, hidden_dim))
        if 'min_expert' in experts_list:
            experts.append(MinPoolingAggregator(input_dim, hidden_dim))
        if 'diffusion_expert' in experts_list:
            experts.append(DiffusionAggregator(input_dim, hidden_dim))
        self.experts = nn.ModuleList(
            [GraphExpert(expert) for expert in experts])
        num_experts = len(experts)

        self.gating = SparseGatingNetwork(input_dim, hidden_dim, num_experts,
                                          num_used_experts)

    def forward(self, x_enc, adj, batch_unknown_nodes, *args):
        gate_weights = self.gating(x_enc)
        expert_outputs = [
            expert(x_enc, adj, batch_unknown_nodes, *args)
            for expert in self.experts
        ]
        expert_outputs_stack = torch.stack(expert_outputs, dim=2)
        combied_out = torch.sum(gate_weights.unsqueeze(-1) *
                                expert_outputs_stack,
                                dim=2)
        return combied_out
