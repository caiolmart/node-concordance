import logging
import os
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from torch_geometric.loader import DataLoader

from src.torch_geo_models import DotMLPRelu


HIDDEN_CHANNELS = 128
IN_CHANNELS = 128
DROPOUT = 0.5
LEARNING_RATIO = 0.005
PREDICTOR_PATH_PAT = 'models/structural_omega_mlp/link_predictor_{run}run_{n_layers}layers_epoch{epoch:04d}.pt'
METRICS_PATH = 'data/metrics/structural_omega_mlp_{n_layers}layers.csv'
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


class StructuralOmegaMLP():

    def __init__(
            self,
            device,
            eval_steps=100,
            n_layers=1,
            epochs=5000,
            batch_size=128 * 1024,
            run=0):
        self.n_layers = n_layers
        self.initialize_models_data(device)
        self.eval_steps = eval_steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.predictor_path_pat = PREDICTOR_PATH_PAT
        self.model_metrics_path = METRICS_PATH
        self.run = run
        self.loss = torch.nn.BCELoss(reduction='mean')

    def initialize_models_data(self, device):

        self.predictor = DotMLPRelu(
            self.n_layers, IN_CHANNELS, HIDDEN_CHANNELS, DROPOUT).to(device)
        self.predictor.reset_parameters()

        self.optimizer = torch.optim.Adam(
            list(self.predictor.parameters()),
            lr=LEARNING_RATIO)

    def train_epoch(self, edges_train, batch_size, x, y):

        self.predictor.train()

        total_loss = total_examples = 0
        for perm in DataLoader(edges_train,
                               batch_size,
                               shuffle=True):

            self.optimizer.zero_grad()
            theta_perm = self.is_same_class(perm, y).to(torch.float32)

            predicted = self.predictor(x[perm[0]], x[perm[1]])

            perm_loss = self.loss(predicted, theta_perm)

            perm_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)

            self.optimizer.step()

            num_examples = perm.size(0)
            total_loss += perm_loss.item() * num_examples
            total_examples += num_examples

        return total_loss / total_examples

    def forward(self, edges, x):
        probas = []
        for batch in DataLoader(edges,
                                self.batch_size,
                                shuffle=False):
            batch_probas = self.predictor(x[batch[0]], x[batch[1]])
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
            y):
        self.predictor.eval()

        losses = []
        aucs = []

        # Loss evaluation
        for edges in [edges_train, edges_val, edges_test]:
            proba = self.forward(edges, x)
            theta = self.is_same_class(edges, y).to(torch.float32)

            loss = self.loss(proba, theta)
            losses.append(loss.item())

            proba = proba.to('cpu').detach().numpy()
            theta = theta.to('cpu').detach().numpy()
            auc = roc_auc_score(theta, proba)
            aucs.append(auc)

        return *losses, *aucs

    def save_models(self, epoch):

        predictor_path = self.predictor_path_pat.format(
            run=self.run,
            n_layers=self.n_layers,
            epoch=epoch)
        predictor_folder = predictor_path.rsplit('/', 1)[0]
        if not os.path.exists(predictor_folder):
            os.makedirs(predictor_folder)

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
            x,
            y):

        loss_train, loss_val, loss_test, auc_train, auc_val, auc_test = self.eval(
            edges_train,
            edges_val,
            edges_test,
            x,
            y)
        self.save_metrics(
            0,
            loss_train,
            loss_val,
            loss_test,
            auc_train,
            auc_val,
            auc_test)

        for epoch in range(1, 1 + self.epochs):
            self.train_epoch(edges_train, self.batch_size, x, y)

            if epoch % self.eval_steps == 0:
                loss_train, loss_val, loss_test, auc_train, auc_val, auc_test = self.eval(
                    edges_train,
                    edges_val,
                    edges_test,
                    x,
                    y)

                logging.info(
                    f'# Layers: {self.n_layers}, '
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
    def read_metrics(n_layers=1):
        metrics_path = METRICS_PATH.format(n_layers=n_layers)
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
