from typing import Tuple
import logging
import os
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from torch_geometric.loader import DataLoader

from src.torch_geo_models import DotMLPRelu, GraphSAGE, LinkPredictor


HIDDEN_CHANNELS = 128
IN_CHANNELS = 128
DROPOUT = 0.5
LEARNING_RATIO = 0.005
DOTMLP_MODEL_PATH_PAT = 'models/structural_omega_grapn_sage_dotmlp/model_{run}run_{n_layers_graph_sage}gslayers_{n_layers_mlp}mlplayers_epoch{epoch:04d}.pt'
DOTMLP_PREDICTOR_PATH_PAT = 'models/structural_omega_grapn_sage_dotmlp/link_predictor_{run}run_{n_layers_graph_sage}gslayers_{n_layers_mlp}mlplayers_epoch{epoch:04d}.pt'
DOTMLP_METRICS_PATH = 'data/metrics/structural_omega_grapn_sage_dotmlp_{n_layers_graph_sage}gslayers_{n_layers_mlp}mlplayers.csv'
COSSIM_MODEL_PATH_PAT = 'models/structural_omega_grapn_sage_cossim/model_{run}run_{n_layers_graph_sage}gslayers_epoch{epoch:04d}.pt'
COSSIM_PREDICTOR_PATH_PAT = 'models/structural_omega_grapn_sage_cossim/link_predictor_{run}run_{n_layers_graph_sage}gslayers_epoch{epoch:04d}.pt'
COSSIM_METRICS_PATH = 'data/metrics/structural_omega_grapn_sage_cossim_{n_layers_graph_sage}gslayers.csv'
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


class StructuralOmegaGraphSageDotMLP():

    def __init__(
            self,
            device,
            eval_steps=100,
            n_layers_graph_sage=1,
            n_layers_mlp=1,
            epochs=5000,
            batch_size=128 * 1024,
            run=0):
        self.n_layers_graph_sage = n_layers_graph_sage
        self.n_layers_mlp = n_layers_mlp
        self.initialize_models_data(device)
        self.eval_steps = eval_steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.predictor_path_pat = DOTMLP_PREDICTOR_PATH_PAT
        self.model_path_pat = DOTMLP_MODEL_PATH_PAT
        self.model_metrics_path = DOTMLP_METRICS_PATH
        self.run = run
        self.loss = torch.nn.BCELoss(reduction='mean')

    def initialize_models_data(self, device):
        self.model = GraphSAGE(self.n_layers_graph_sage,
                               IN_CHANNELS,
                               HIDDEN_CHANNELS,
                               DROPOUT).to(device)
        self.model.reset_parameters()

        self.predictor = DotMLPRelu(
            self.n_layers_mlp,
            HIDDEN_CHANNELS,
            HIDDEN_CHANNELS,
            DROPOUT).to(device)
        self.predictor.reset_parameters()

        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) +
            list(self.predictor.parameters()),
            lr=LEARNING_RATIO)

    def train_epoch(self, edges_train, batch_size, x, y, adj_t):

        self.predictor.train()

        total_loss = total_examples = 0
        for perm in DataLoader(edges_train,
                               batch_size,
                               shuffle=True):

            self.optimizer.zero_grad()
            theta_perm = self.is_same_class(perm, y).to(torch.float32)

            h = self.model(x, adj_t)
            predicted = self.predictor(h[perm[0]], h[perm[1]])

            perm_loss = self.loss(predicted, theta_perm)

            perm_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)

            self.optimizer.step()

            num_examples = perm.size(0)
            total_loss += perm_loss.item() * num_examples
            total_examples += num_examples

        return total_loss / total_examples

    def forward(self, edges, x, adj_t):
        probas = []
        for batch in DataLoader(edges,
                                self.batch_size,
                                shuffle=False):

            h = self.model(x, adj_t)
            batch_probas = self.predictor(h[batch[0]], h[batch[1]])
            probas.append(batch_probas)
        return torch.concat(probas)

    def is_same_class(self, edges, y):
        return (y[edges[0]] == y[edges[1]]) * 1

    def eval(
            self,
            edges_train,
            edges_val,
            edges_test,
            x,
            y,
            adj_t):
        self.predictor.eval()

        losses = []
        aucs = []

        # Loss evaluation
        for edges in [edges_train, edges_val, edges_test]:
            proba = self.forward(edges, x, adj_t)
            theta = self.is_same_class(edges, y).to(torch.float32)

            loss = self.loss(proba, theta)
            losses.append(loss.item())

            proba = proba.to('cpu').detach().numpy()
            theta = theta.to('cpu').detach().numpy()
            auc = roc_auc_score(theta, proba)
            aucs.append(auc)

        return *losses, *aucs

    def save_models(self, epoch):

        model_path = self.model_path_pat.format(
            run=self.run,
            n_layers_graph_sage=self.n_layers_graph_sage,
            n_layers_mlp=self.n_layers_mlp,
            epoch=epoch)
        model_folder = model_path.rsplit('/', 1)[0]
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        predictor_path = self.predictor_path_pat.format(
            run=self.run,
            n_layers_graph_sage=self.n_layers_graph_sage,
            n_layers_mlp=self.n_layers_mlp,
            epoch=epoch)
        predictor_folder = predictor_path.rsplit('/', 1)[0]
        if not os.path.exists(predictor_folder):
            os.makedirs(predictor_folder)

        torch.save(self.model.state_dict(), model_path)
        torch.save(self.predictor.state_dict(), predictor_path)

    def save_metrics(
            self,
            epoch,
            loss_train,
            loss_val,
            loss_test,
            auc_train,
            auc_val,
            auc_test):

        metrics_path = self.model_metrics_path.format(
            n_layers_graph_sage=self.n_layers_graph_sage,
            n_layers_mlp=self.n_layers_mlp,
        )
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
            x,
            y,
            adj_t):

        loss_train, loss_val, loss_test, auc_train, auc_val, auc_test = self.eval(
            edges_train,
            edges_val,
            edges_test,
            x,
            y,
            adj_t)
        self.save_metrics(
            0,
            loss_train,
            loss_val,
            loss_test,
            auc_train,
            auc_val,
            auc_test)

        for epoch in range(1, 1 + self.epochs):
            self.train_epoch(edges_train, self.batch_size, x, y, adj_t)

            if epoch % self.eval_steps == 0:
                loss_train, loss_val, loss_test, auc_train, auc_val, auc_test = self.eval(
                    edges_train,
                    edges_val,
                    edges_test,
                    x,
                    y,
                    adj_t)

                logging.info(
                    f'# Graph Sage Layers: {self.n_layers_graph_sage}, '
                    f'# MLP Layers: {self.n_layers_mlp}, '
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
                    loss_val,
                    loss_test,
                    auc_train,
                    auc_val,
                    auc_test)

    @staticmethod
    def read_metrics(n_layers_graph_sage=1, n_layers_mlp=1):
        metrics_path = DOTMLP_METRICS_PATH.format(
            n_layers_graph_sage=n_layers_graph_sage,
            n_layers_mlp=n_layers_mlp)
        return pd.read_csv(metrics_path)

    @classmethod
    def load_model(
        cls,
        run,
        epoch,
        device,
        num_nodes,
        eval_steps=100,
        n_layers=1,
        epochs=5000,
        batch_size=128 * 1024
    ):

        omega = cls(
            device,
            num_nodes,
            run=run,
            eval_steps=eval_steps,
            n_layers=n_layers,
            epochs=epochs,
            batch_size=batch_size)

        predictor_path = omega.predictor_path_pat.format(
            run=omega.run,
            n_layers=omega.n_layers,
            epoch=epoch)

        predictor = DotMLPRelu().to(device)
        predictor.load_state_dict(torch.load(predictor_path))
        predictor.eval()

        omega.predictor = predictor

        return omega


class StructuralOmegaGraphSageCosSim():

    def __init__(
            self,
            device,
            eval_steps=100,
            n_layers_graph_sage=1,
            epochs=5000,
            batch_size=128 * 1024,
            run=0):
        self.n_layers_graph_sage = n_layers_graph_sage
        self.initialize_models_data(device)
        self.eval_steps = eval_steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.predictor_path_pat = COSSIM_PREDICTOR_PATH_PAT
        self.model_path_pat = COSSIM_MODEL_PATH_PAT
        self.model_metrics_path = COSSIM_METRICS_PATH
        self.run = run
        self.loss = torch.nn.BCELoss(reduction='mean')

    def initialize_models_data(self, device):
        self.model = GraphSAGE(self.n_layers_graph_sage,
                               IN_CHANNELS,
                               HIDDEN_CHANNELS,
                               DROPOUT).to(device)
        self.model.reset_parameters()

        self.predictor = LinkPredictor().to(device)
        self.predictor.reset_parameters()

        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) +
            list(self.predictor.parameters()),
            lr=LEARNING_RATIO)

    def train_epoch(self, edges_train, batch_size, x, y, adj_t):

        self.predictor.train()

        total_loss = total_examples = 0
        for perm in DataLoader(edges_train,
                               batch_size,
                               shuffle=True):

            self.optimizer.zero_grad()
            theta_perm = self.is_same_class(perm, y).to(torch.float32)

            h = self.model(x, adj_t)
            predicted = self.predictor(h[perm[0]], h[perm[1]])

            perm_loss = self.loss(predicted, theta_perm)

            perm_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)

            self.optimizer.step()

            num_examples = perm.size(0)
            total_loss += perm_loss.item() * num_examples
            total_examples += num_examples

        return total_loss / total_examples

    def forward(self, edges, x, adj_t):
        probas = []
        for batch in DataLoader(edges,
                                self.batch_size,
                                shuffle=False):

            h = self.model(x, adj_t)
            batch_probas = self.predictor(h[batch[0]], h[batch[1]])
            probas.append(batch_probas)
        return torch.concat(probas)

    def is_same_class(self, edges, y):
        return (y[edges[0]] == y[edges[1]]) * 1

    def eval(
            self,
            edges_train,
            edges_val,
            edges_test,
            x,
            y,
            adj_t):
        self.predictor.eval()

        losses = []
        aucs = []

        # Loss evaluation
        for edges in [edges_train, edges_val, edges_test]:
            proba = self.forward(edges, x, adj_t)
            theta = self.is_same_class(edges, y).to(torch.float32)

            loss = self.loss(proba, theta)
            losses.append(loss.item())

            proba = proba.to('cpu').detach().numpy()
            theta = theta.to('cpu').detach().numpy()
            auc = roc_auc_score(theta, proba)
            aucs.append(auc)

        return *losses, *aucs

    def save_models(self, epoch):

        model_path = self.model_path_pat.format(
            run=self.run,
            n_layers_graph_sage=self.n_layers_graph_sage,
            epoch=epoch)
        model_folder = model_path.rsplit('/', 1)[0]
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        predictor_path = self.predictor_path_pat.format(
            run=self.run,
            n_layers_graph_sage=self.n_layers_graph_sage,
            epoch=epoch)
        predictor_folder = predictor_path.rsplit('/', 1)[0]
        if not os.path.exists(predictor_folder):
            os.makedirs(predictor_folder)

        torch.save(self.model.state_dict(), model_path)
        torch.save(self.predictor.state_dict(), predictor_path)

    def save_metrics(
            self,
            epoch,
            loss_train,
            loss_val,
            loss_test,
            auc_train,
            auc_val,
            auc_test):

        metrics_path = self.model_metrics_path.format(
            n_layers_graph_sage=self.n_layers_graph_sage,
        )
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
            x,
            y,
            adj_t):

        loss_train, loss_val, loss_test, auc_train, auc_val, auc_test = self.eval(
            edges_train,
            edges_val,
            edges_test,
            x,
            y,
            adj_t)
        self.save_metrics(
            0,
            loss_train,
            loss_val,
            loss_test,
            auc_train,
            auc_val,
            auc_test)

        for epoch in range(1, 1 + self.epochs):
            self.train_epoch(edges_train, self.batch_size, x, y, adj_t)

            if epoch % self.eval_steps == 0:
                loss_train, loss_val, loss_test, auc_train, auc_val, auc_test = self.eval(
                    edges_train,
                    edges_val,
                    edges_test,
                    x,
                    y,
                    adj_t)

                logging.info(
                    f'# Graph Sage Layers: {self.n_layers_graph_sage}, '
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
                    loss_val,
                    loss_test,
                    auc_train,
                    auc_val,
                    auc_test)

    @staticmethod
    def read_metrics(n_layers_graph_sage=1, n_layers_mlp=1):
        metrics_path = COSSIM_METRICS_PATH.format(
            n_layers_graph_sage=n_layers_graph_sage)
        return pd.read_csv(metrics_path)

    @classmethod
    def load_model(
        cls,
        run,
        epoch,
        device,
        num_nodes,
        eval_steps=100,
        n_layers=1,
        epochs=5000,
        batch_size=128 * 1024
    ):

        omega = cls(
            device,
            num_nodes,
            run=run,
            eval_steps=eval_steps,
            n_layers=n_layers,
            epochs=epochs,
            batch_size=batch_size)

        model_path = omega.model_path_pat.format(
            run=omega.run,
            n_layers=omega.n_layers,
            epoch=epoch)

        model = GraphSAGE(
            omega.n_layers,
            IN_CHANNELS,
            HIDDEN_CHANNELS,
            DROPOUT).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        predictor_path = omega.predictor_path_pat.format(
            run=omega.run,
            n_layers_graph_sage=omega.n_layers_graph_sage,
            epoch=epoch)

        predictor = LinkPredictor().to(device)
        predictor.load_state_dict(torch.load(predictor_path))
        predictor.eval()

        omega.model = model
        omega.predictor = predictor

        return omega