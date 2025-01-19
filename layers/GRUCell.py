from typing import Tuple, Union

import torch
from torch import Tensor, nn

StateType = Union[Tensor, Tuple[Tensor]]


class GRUCell(nn.Module):
    """Base class for implementing gated recurrent unit (GRU) cells."""

    def __init__(self, hidden_size: int, forget_gate: nn.Module,
                 update_gate: nn.Module, candidate_gate: nn.Module):
        super().__init__()
        self.hidden_size = hidden_size
        self.forget_gate = forget_gate
        self.update_gate = update_gate
        self.candidate_gate = candidate_gate

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(hidden_size={self.hidden_size})'

    def reset_parameters(self):
        self.forget_gate.reset_parameters()
        self.update_gate.reset_parameters()
        self.candidate_gate.reset_parameters()

    def initialize_state(self, x) -> Tensor:
        return torch.zeros(x.size(0),
                           self.hidden_size,
                           dtype=x.dtype,
                           device=x.device)

    def forward(self, x: Tensor, h: Tensor, *args, **kwargs) -> Tensor:
        """"""
        # x: [batch, *, channels]
        # h: [batch, *, channels]
        x_gates = torch.cat([x, h], dim=-1)
        r = torch.sigmoid(self.forget_gate(x_gates, *args, **kwargs))
        u = torch.sigmoid(self.update_gate(x_gates, *args, **kwargs))
        x_c = torch.cat([x, r * h], dim=-1)
        c = torch.tanh(self.candidate_gate(x_c, *args, **kwargs))
        h_new = u * h + (1. - u) * c
        return h_new


class GraphGRUCellBase(GRUCell):
    """Base class for implementing graph-based gated recurrent unit (GRU)
    cells."""

    def initialize_state(self, x) -> Tensor:
        return torch.zeros(x.size(0),
                           x.size(-2),
                           self.hidden_size,
                           dtype=x.dtype,
                           device=x.device)


class MoGEGRUCellBase(GRUCell):
    """Base class for implementing graph-based gated recurrent unit (GRU)
    cells."""

    def initialize_state(self, x) -> Tensor:
        return torch.zeros(x.size(0),
                           x.size(-2),
                           self.hidden_size,
                           dtype=x.dtype,
                           device=x.device)

    def forward(self, x: Tensor, h: Tensor, adj, bacth_unknown_nodes, A_q,
                A_h) -> Tensor:
        """"""
        # x: [batch, *, channels]
        # h: [batch, *, channels]
        x_gates = torch.cat([x, h], dim=-1)
        r = torch.sigmoid(
            self.forget_gate(x_gates, adj, bacth_unknown_nodes, A_q, A_h))
        u = torch.sigmoid(
            self.update_gate(x_gates, adj, bacth_unknown_nodes, A_q, A_h))
        x_c = torch.cat([x, r * h], dim=-1)
        c = torch.tanh(
            self.candidate_gate(x_c, adj, bacth_unknown_nodes, A_q, A_h))
        h_new = u * h + (1. - u) * c
        return h_new
