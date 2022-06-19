import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv


class GraphSAGE(torch.nn.Module):
    def __init__(self, n_layers, in_channels, out_channels, dropout):
        super(GraphSAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, out_channels))
        for _ in range(n_layers - 1):
            self.convs.append(SAGEConv(out_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, bias=True):
        super(LinkPredictor, self).__init__()
        self.lin = torch.nn.Linear(1, 1, bias=bias)

    def reset_parameters(self):
        self.lin.weight.data.fill_(1)
        self.lin.bias.data.fill_(0)

    def forward(self, x_i, x_j):
        cos_sim = torch.sum(
            torch.mul(F.normalize(x_i), 
                      F.normalize(x_j)), 
            dim=1,
            keepdim=True
        )
        x = self.lin(cos_sim)
        return torch.sigmoid(x)


class GCN(torch.nn.Module):
    def __init__(
            self,
            n_layers,
            in_channels,
            hidden_channels,
            out_channels,
            dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()

        if n_layers == 1:
            self.convs.append(GCNConv(in_channels, out_channels))
        else:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            for _ in range(n_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.convs.append(GCNConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t, edge_weight=None):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)
