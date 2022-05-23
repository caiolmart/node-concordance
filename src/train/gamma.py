import logging
import os
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling

from src.torch_geo_models import GraphSAGE, LinkPredictor


HIDDEN_CHANNELS = 50
DROPOUT = 0.5
LEARNING_RATIO = 0.005
MODEL_PATH_PAT = 'models/gamma/graph_sage/{run}run_{n_layers}layers_epoch{epoch:04d}.pt'
PREDICTOR_PATH_PAT = 'models/gamma/link_predictor/{run}run_{n_layers}layers_epoch{epoch:04d}.pt'
METRICS_PATH = 'data/metrics/gamma_graph_sage_{n_layers}layers.csv'
METRICS_COLS = [
    'run',
    'epoch',
    'loss_train',
    'loss_val',
    'loss_test',
    'auc_train',
    'auc_val',
    'auc_test',
]


class GammaGraphSage():

    def __init__(
            self,
            device,
            num_nodes,
            eval_steps=1,
            n_layers=1,
            epochs=100,
            batch_size=128 * 1024,
            run=0):
        self.n_layers = n_layers
        self.initialize_models_data(device, num_nodes)
        self.eval_steps = eval_steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_path_pat = MODEL_PATH_PAT
        self.predictor_path_pat = PREDICTOR_PATH_PAT
        self.model_metrics_path = METRICS_PATH
        self.run = run

    def initialize_models_data(self, device, num_nodes):

        self.model = GraphSAGE(self.n_layers,
                               HIDDEN_CHANNELS,
                               HIDDEN_CHANNELS,
                               DROPOUT).to(device)
        self.model.reset_parameters()
        self.predictor = LinkPredictor().to(device)
        self.predictor.reset_parameters()
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

    def forward(self, edges, adj_t):
        self.model.eval()
        self.predictor.eval()
        emb_weight = self.embedding.weight

        h = self.model(emb_weight, adj_t)
        probas = self.predictor(h[edges[0]], h[edges[1]])

        return probas

    def is_same_class(self, edges, y):
        return (y[edges[0]] == y[edges[1]]) * 1

    def eval(
            self,
            edges_train,
            edges_val,
            edges_test,
            neg_edges_val,
            neg_edges_test,
            adj_t,
            y):

        proba_pos_train = self.forward(edges_train, adj_t)
        proba_pos_val = self.forward(edges_val, adj_t)
        proba_pos_test = self.forward(edges_test, adj_t)

        proba_neg_val = self.forward(neg_edges_val, adj_t)
        proba_neg_test = self.forward(neg_edges_test, adj_t)

        # Loss evaluation

        val_pos_loss = -torch.log(proba_pos_val + 1e-15).mean()
        val_neg_loss = -torch.log(1 - proba_neg_val + 1e-15).mean()

        test_pos_loss = -torch.log(proba_pos_test + 1e-15).mean()
        test_neg_loss = -torch.log(1 - proba_neg_test + 1e-15).mean()

        loss_val = val_pos_loss + val_neg_loss
        loss_test = test_pos_loss + test_neg_loss

        # AUC evaluation

        proba_pos_train = proba_pos_train\
            .to('cpu')\
            .detach()\
            .numpy()
        proba_pos_val = proba_pos_val\
            .to('cpu')\
            .detach()\
            .numpy()
        proba_pos_test = proba_pos_test\
            .to('cpu')\
            .detach()\
            .numpy()

        theta_train = self.is_same_class(edges_train, y)\
            .to('cpu')\
            .detach()\
            .numpy()
        theta_val = self.is_same_class(edges_val, y)\
            .to('cpu')\
            .detach()\
            .numpy()
        theta_test = self.is_same_class(edges_test, y)\
            .to('cpu')\
            .detach()\
            .numpy()

        auc_train = roc_auc_score(theta_train, proba_pos_train)
        auc_val = roc_auc_score(theta_val, proba_pos_val)
        auc_test = roc_auc_score(theta_test, proba_pos_test)

        return loss_val, loss_test, auc_train, auc_val, auc_test

    def save_models(self, epoch):
        model_path = self.model_path_pat.format(
            run=self.run,
            n_layers=self.n_layers,
            epoch=epoch)
        model_folder = model_path.rsplit('/', 1)[0]
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        predictor_path = self.predictor_path_pat.format(
            run=self.run,
            n_layers=self.n_layers,
            epoch=epoch)
        predictor_folder = predictor_path.rsplit('/', 1)[0]
        if not os.path.exists(predictor_folder):
            os.makedirs(predictor_folder)
        torch.save(self.model, model_path)
        torch.save(self.predictor, predictor_path)

    def save_metrics(
            self,
            epoch,
            loss_train,
            loss_val,
            loss_test,
            auc_train,
            auc_val,
            auc_test):

        metrics_path = self.model_metrics_path.format(n_layers=self.n_layers)
        metrics_folder = metrics_path.rsplit('/', 1)[0]
        if not os.path.exists(metrics_folder):
            os.makedirs(metrics_folder)
        if not os.path.exists(metrics_path):
            header = ','.join(METRICS_COLS) + '\n'
            with open(metrics_path, 'w') as stream:
                stream.write(header)

        metrics = [
            self.run,
            epoch,
            loss_train,
            loss_val,
            loss_test,
            auc_train,
            auc_val,
            auc_test
        ]

        metrics_row = ','.join(str(x) for x in metrics) + '\n'
        with open(metrics_path, 'a') as stream:
            stream.write(metrics_row)

    def train(
            self,
            edges_train,
            edges_val,
            edges_test,
            neg_edges_val,
            neg_edges_test,
            adj_t,
            y):

        loss_val, loss_test, auc_train, auc_val, auc_test = self.eval(
            edges_train,
            edges_val,
            edges_test,
            neg_edges_val,
            neg_edges_test,
            adj_t,
            y)
        self.save_metrics(
            0,
            None,
            loss_val.item(),
            loss_test.item(),
            auc_train,
            auc_val,
            auc_test)

        for epoch in range(1, 1 + self.epochs):
            loss_train = self.train_epoch(
                edges_train,
                self.batch_size,
                adj_t)

            if epoch % self.eval_steps == 0:
                loss_val, loss_test, auc_train, auc_val, auc_test = self.eval(
                    edges_train,
                    edges_val,
                    edges_test,
                    neg_edges_val,
                    neg_edges_test,
                    adj_t,
                    y)

                logging.info(
                    f'Run: {self.run:04d}, '
                    f'Epoch: {epoch:04d}, '
                    f'Train Loss: {loss_train:.4f}, '
                    f'Valid loss: {loss_val:.4f}, '
                    f'Test loss: {loss_test:.4f}, '
                    f'Train AUC: {auc_train:.4f}, '
                    f'Valid AUC: {auc_val:.4f}, '
                    f'Test AUC: {auc_test:.4f}')

                self.save_models(epoch)

                self.save_metrics(
                    epoch,
                    loss_train,
                    loss_val.item(),
                    loss_test.item(),
                    auc_train,
                    auc_val,
                    auc_test)

    def read_metrics(self):
        metrics_path = self.model_metrics_path.format(n_layers=self.n_layers)
        return pd.read_csv(metrics_path)
