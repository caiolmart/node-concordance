import torch
import torch.nn.functional as F
from dgl.nn.pytorch.conv import SAGEConv


class GraphSAGE(torch.nn.Module):
    def __init__(
            self,
            n_layers,
            in_channels,
            hidden_channels,
            out_channels,
            dropout,
            aggregator_type='mean'):
        super(GraphSAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        if n_layers == 1:
            self.convs.append(
                SAGEConv(in_channels, out_channels, aggregator_type))
        else:
            self.convs.append(
                SAGEConv(in_channels, hidden_channels, aggregator_type))
            for _ in range(n_layers - 2):
                self.convs.append(
                    SAGEConv(
                        hidden_channels,
                        hidden_channels,
                        aggregator_type))
            self.convs.append(
                SAGEConv(hidden_channels, out_channels, aggregator_type))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, graph, x, edge_weight=None):
        for conv in self.convs[:-1]:
            x = conv(graph, x, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](graph, x, edge_weight=edge_weight)
        return x.log_softmax(dim=-1)


class EdgeWeigher(torch.nn.Module):
    def __init__(
            self,
            base_weights,
            bias=True):
        super(EdgeWeigher, self).__init__()

        self.base_weights = base_weights
        self.lin = torch.nn.Linear(1, 1, bias=bias)

    def reset_parameters(self):
        self.lin.weight.data.fill_(1)
        self.lin.bias.data.fill_(0)

    def forward(self):
        edge_weights = self.lin(self.base_weights)
        edge_weights = torch.sigmoid(edge_weights)

        return edge_weights
