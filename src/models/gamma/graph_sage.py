import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class OneLayerSAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels,
                 dropout):
        super(OneLayerSAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, out_channels))

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
    def __init__(self, out_channels, bias=True):
        super(LinkPredictor, self).__init__()
        self.lin = torch.nn.Linear(1, out_channels, bias=bias)

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
