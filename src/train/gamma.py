import torch
from torch_geometric.data import DataLoader
from torch_geometric.utils import negative_sampling

from src.models import GraphSAGE, LinkPredictor


HIDDEN_CHANNELS = 50
DROPOUT = 0.5
BATCH_SIZE = 128 * 1024
LEARNING_RATIO = 0.005
N_LAYERS = 1
EPOCHS = 300
EVAL_STEPS = 5
RUNS = 10


class GammaGraphSage():

    def __init__(self, device, num_nodes):
        self.initialize_models_data(device, num_nodes)

    def initialize_models_data(self, device, num_nodes):

        self.model = GraphSAGE(N_LAYERS,
                               HIDDEN_CHANNELS,
                               HIDDEN_CHANNELS,
                               DROPOUT).to(device)

        self.predictor = LinkPredictor().to(device)

        self.embedding = torch.nn.Embedding(
            num_nodes,
            HIDDEN_CHANNELS).to(device)
        torch.nn.init.xavier_uniform_(self.embedding.weight)

        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) +
            list(self.embedding.parameters()) +
            list(self.predictor.parameters()),
            lr=LEARNING_RATIO)

    def train_epoch(self, edge_index, batch_size, adj_t):

        self.model.train()
        self.predictor.train()
        emb_weight = self.embedding.weight

        pos_train_edge = edge_index.t().to(emb_weight.device)

        total_loss = total_examples = 0
        for perm in DataLoader(range(pos_train_edge.size(0)),
                               batch_size,
                               shuffle=True):

            self.optimizer.zero_grad()

            h = self.model(emb_weight, adj_t)

            edge = pos_train_edge[perm].t()

            pos_out = self.predictor(h[edge[0]], h[edge[1]])
            pos_loss = -torch.log(pos_out + 1e-15).mean()

            edge = negative_sampling(edge_index,
                                     num_nodes=emb_weight.size(0),
                                     num_neg_samples=perm.size(0),
                                     method='sparse')

            neg_out = self.predictor(h[edge[0]], h[edge[1]])
            neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

            loss = pos_loss + neg_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(emb_weight, 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)

            self.optimizer.step()

            num_examples = pos_out.size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples

        return total_loss / total_examples

    def predict(self, edges, adj_t):
        self.model.eval()
        self.predictor.eval()
        emb_weight = self.embedding.weight
        
        h = self.model(emb_weight, adj_t)
        probas = self.predictor(h[edges[0]], h[edges[1]])
        
        return probas

    def is_same_class(self, edges, y):
        return (y[edges[0]] == y[edges[1]]) * 1
