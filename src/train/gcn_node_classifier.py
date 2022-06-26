import logging
import os
import pandas as pd
import torch
import torch.nn.functional as F

from src.data.node_classifier.arxiv import get_edge_weights_pyg
from src.torch_geo_models import GCN


DROPOUT = 0.5
LEARNING_RATIO = 0.001
N_LAYERS = list(range(1, 5))
EPOCHS = 5000
LOG_STEPS = 50
DROPOUT = 0.5

MODEL_PATH_PAT = 'models/node_classifier/gcn/{run:02d}run_{weighed}weights_{n_layers}layers_epoch{epoch:04d}.pt'
METRICS_PATH = 'data/metrics/gcn_node_classifier.csv'
METRICS_COLS = [
    'run',
    'epoch',
    'has_edge_weights',
    'n_layers',
    'loss_train',
    'loss_val',
    'loss_test',
    'acc_train',
    'acc_val',
    'acc_test',
]
BOOL_STR_MAP = {
    True: 'with',
    False: 'no'
}



class GCNNodeClassifierTrainer():

    def __init__(
            self,
            device,
            evaluator,
            n_layers=1,
            input_dim=128,
            hidden_channels=128 * 2,
            output_dim=40,
            edge_weights=None,
            run=0):
        
        self.n_layers = n_layers
        self.device = device
        self.evaluator = evaluator
        self.edge_weights = edge_weights
        self.has_edge_weights = False
        if edge_weights is not None:
            self.has_edge_weights = True
        self.input_dim = input_dim
        self.hidden_channels=hidden_channels
        self.output_dim = output_dim
        self.dropout = DROPOUT
        self.initialize_model()
        self.eval_steps = LOG_STEPS
        self.epochs = EPOCHS
        self.model_path_pat = MODEL_PATH_PAT
        self.model_metrics_path = METRICS_PATH
        self.run = run

    def initialize_model(self):

        self.model = GCN(
            n_layers=self.n_layers,
            in_channels=self.input_dim,
            hidden_channels=self.hidden_channels,
            out_channels=self.output_dim,
            dropout=self.dropout,
            batch_norm=True)\
            .to(self.device)
        self.model.reset_parameters()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=LEARNING_RATIO)

    def train_epoch(
            self,
            features,
            adj_t,
            labels,
            train_mask):

        self.model.train()

        self.optimizer.zero_grad()
        out = self.model(features, adj_t, edge_weight=self.edge_weights)[train_mask]
        loss = F.nll_loss(out, labels.squeeze(1)[train_mask])
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def eval(
            self,
            features,
            adj_t,
            labels,
            train_mask,
            val_mask,
            test_mask):

        self.model.eval()

        out = self.model(features, adj_t, edge_weight=self.edge_weights)
        y_pred = out.argmax(dim=-1, keepdim=True)

        acc_train = self.evaluator.eval({
            'y_true': labels[train_mask],
            'y_pred': y_pred[train_mask],
        })['acc']
        acc_val = self.evaluator.eval({
            'y_true': labels[val_mask],
            'y_pred': y_pred[val_mask],
        })['acc']
        acc_test = self.evaluator.eval({
            'y_true': labels[test_mask],
            'y_pred': y_pred[test_mask],
        })['acc']

        loss_train = F.nll_loss(
            out[train_mask],
            labels.squeeze(1)[train_mask]).item()
        loss_val = F.nll_loss(
            out[val_mask],
            labels.squeeze(1)[val_mask]).item()
        loss_test = F.nll_loss(
            out[test_mask],
            labels.squeeze(1)[test_mask]).item()

        return acc_train, acc_val, acc_test, loss_train, loss_val, loss_test

    def save_model(self, epoch):
        model_path = self.model_path_pat.format(
            run=self.run,
            n_layers=self.n_layers,
            epoch=epoch,
            weighed=BOOL_STR_MAP[self.has_edge_weights])
        model_folder = model_path.rsplit('/', 1)[0]
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        torch.save(self.model.state_dict(), model_path)

    def save_metrics(
            self,
            epoch,
            acc_train,
            acc_val,
            acc_test,
            loss_train,
            loss_val,
            loss_test):
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
            self.has_edge_weights,
            self.n_layers,
            loss_train,
            loss_val,
            loss_test,
            acc_train,
            acc_val,
            acc_test,
        ]

        metrics_row = ','.join(str(x) for x in metrics) + '\n'
        with open(metrics_path, 'a') as stream:
            stream.write(metrics_row)

    def train(
            self,
            features,
            adj_t,
            labels,
            train_mask,
            val_mask,
            test_mask):

        acc_train, acc_val, acc_test, loss_train, loss_val, loss_test =\
            self.eval(
                features,
                adj_t,
                labels,
                train_mask,
                val_mask,
                test_mask)
        self.save_metrics(
            0,
            acc_train,
            acc_val,
            acc_test,
            loss_train,
            loss_val,
            loss_test)

        for epoch in range(1, 1 + self.epochs):
            loss_train = self.train_epoch(
                features,
                adj_t,
                labels,
                train_mask)

            if epoch % self.eval_steps == 0:
                acc_train, acc_val, acc_test, loss_train, loss_val, loss_test =\
                    self.eval(
                        features,
                        adj_t,
                        labels,
                        train_mask,
                        val_mask,
                        test_mask)

                logging.info(
                    f'Run: {self.run:04d}, '
                    f'Epoch: {epoch:04d}, '
                    f'{self.n_layers} layers, '
                    f'{BOOL_STR_MAP[self.has_edge_weights]} weights, '
                    f'Train Loss: {loss_train:.4f}, '
                    f'Valid loss: {loss_val:.4f}, '
                    f'Test loss: {loss_test:.4f}, '
                    f'Train ACC: {acc_train:.4f}, '
                    f'Valid ACC: {acc_val:.4f}, '
                    f'Test ACC: {acc_test:.4f}')

                self.save_model(epoch)

                self.save_metrics(
                    epoch,
                    acc_train,
                    acc_val,
                    acc_test,
                    loss_train,
                    loss_val,
                    loss_test)

    @classmethod
    def read_metrics(self):
        metrics_path = METRICS_PATH
        return pd.read_csv(metrics_path)
