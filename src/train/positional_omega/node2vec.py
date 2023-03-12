import logging
import os
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from torch_geometric.nn.models import Node2Vec

from src.torch_geo_models import LinkPredictor


EMBEDDING_DIM = 50
WALK_LENGTH = 20
CONTEXT_SIZE = 10
WALKS_PER_NODE = 1
DROPOUT = 0.5
LEARNING_RATIO = 0.01
PREDICTOR_PATH_PAT = 'models/positional_omega_node2vec/link_predictor/{run}run_{p}p_{q}q_epoch{epoch:04d}.pt'
EMBEDDING_PATH_PAT = 'models/positional_omega_node2vec/embedding/{run}run_{p}p_{q}q_epoch{epoch:04d}.pt'
METRICS_PATH = 'data/metrics/{dataset}/positional_omega_node2vec_{p}p_{q}q.csv'
METRICS_COLS = [
    'p',
    'q',
    'run',
    'epoch',
    'node2vec_loss',
    'loss_train',
    'loss_val',
    'loss_test',
    'auc_train',
    'auc_val',
    'auc_test',
]


class PositionalOmegaNode2Vec():

    def __init__(
            self,
            device,
            edge_index,
            num_nodes,
            p,
            q,
            eval_steps=10,
            epochs=500,
            batch_size=8 * 1024,
            run=0):
        self.device = device
        self.p = p
        self.q = q
        self.batch_size = batch_size
        self.initialize_models_data(device, edge_index, num_nodes)
        self.eval_steps = eval_steps
        self.epochs = epochs
        self.predictor_path_pat = PREDICTOR_PATH_PAT
        self.embedding_path_pat = EMBEDDING_PATH_PAT
        self.model_metrics_path = METRICS_PATH
        self.run = run

    def initialize_models_data(self, device, edge_index, num_nodes):

        self.model = Node2Vec(edge_index,
                              EMBEDDING_DIM,
                              WALK_LENGTH,
                              CONTEXT_SIZE,
                              walks_per_node=WALKS_PER_NODE,
                              p=self.p,
                              q=self.q,
                              num_negative_samples=WALKS_PER_NODE,
                              num_nodes=num_nodes).to(device)
        self.model.reset_parameters()

        self.loader = self.model.loader(
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1
        )

        self.predictor = LinkPredictor().to(device)
        self.predictor.reset_parameters()

        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) +
            list(self.predictor.parameters()),
            lr=LEARNING_RATIO)

    def train_epoch(self):

        self.model.train()
        self.predictor.train()

        total_loss = total_examples = 0
        for pos_rw, neg_rw in self.loader:

            self.optimizer.zero_grad()

            loss = self.model.loss(
                pos_rw.to(self.device),
                neg_rw.to(self.device))

            loss.backward()
            self.optimizer.step()

            num_examples = pos_rw.size(0)
            total_loss += loss * num_examples
            total_examples += num_examples

        return total_loss / total_examples

    def forward(self, edges):
        h = self.model.embedding.weight
        probas = self.predictor(h[edges[0]], h[edges[1]])

        return probas

    def is_same_class(self, edges, y):
        return (y[edges[0]] == y[edges[1]]) * 1

    def eval(
            self,
            edges_train,
            edges_val,
            edges_test,
            neg_edges_train,
            neg_edges_val,
            neg_edges_test,
            edges_train_auc,
            edges_val_auc,
            edges_test_auc,
            y):
        self.model.eval()
        self.predictor.eval()

        proba_pos_train = self.forward(edges_train)
        proba_pos_val = self.forward(edges_val)
        proba_pos_test = self.forward(edges_test)

        proba_neg_train = self.forward(neg_edges_train)
        proba_neg_val = self.forward(neg_edges_val)
        proba_neg_test = self.forward(neg_edges_test)

        # Loss evaluation

        train_pos_loss = -torch.log(proba_pos_train + 1e-15).mean()
        train_neg_loss = -torch.log(1 - proba_neg_train + 1e-15).mean()

        val_pos_loss = -torch.log(proba_pos_val + 1e-15).mean()
        val_neg_loss = -torch.log(1 - proba_neg_val + 1e-15).mean()

        test_pos_loss = -torch.log(proba_pos_test + 1e-15).mean()
        test_neg_loss = -torch.log(1 - proba_neg_test + 1e-15).mean()

        loss_train = train_pos_loss + train_neg_loss
        loss_val = val_pos_loss + val_neg_loss
        loss_test = test_pos_loss + test_neg_loss

        # AUC evaluation

        proba_pos_train = self.forward(edges_train_auc)\
            .to('cpu')\
            .detach()\
            .numpy()
        proba_pos_val = self.forward(edges_val_auc)\
            .to('cpu')\
            .detach()\
            .numpy()
        proba_pos_test = self.forward(edges_test_auc)\
            .to('cpu')\
            .detach()\
            .numpy()

        theta_train = self.is_same_class(edges_train_auc, y)\
            .to('cpu')\
            .detach()\
            .numpy()
        theta_val = self.is_same_class(edges_val_auc, y)\
            .to('cpu')\
            .detach()\
            .numpy()
        theta_test = self.is_same_class(edges_test_auc, y)\
            .to('cpu')\
            .detach()\
            .numpy()

        auc_train = roc_auc_score(theta_train, proba_pos_train)
        auc_val = roc_auc_score(theta_val, proba_pos_val)
        auc_test = roc_auc_score(theta_test, proba_pos_test)

        return loss_train, loss_val, loss_test, auc_train, auc_val, auc_test

    def save_models(self, epoch):

        predictor_path = self.predictor_path_pat.format(
            run=self.run,
            p=self.p,
            q=self.q,
            epoch=epoch)
        predictor_folder = predictor_path.rsplit('/', 1)[0]
        if not os.path.exists(predictor_folder):
            os.makedirs(predictor_folder)

        embedding_path = self.embedding_path_pat.format(
            run=self.run,
            p=self.p,
            q=self.q,
            epoch=epoch)
        embedding_folder = embedding_path.rsplit('/', 1)[0]
        if not os.path.exists(embedding_folder):
            os.makedirs(embedding_folder)

        torch.save(self.predictor.state_dict(), predictor_path)
        torch.save(self.model.state_dict(), embedding_path)

    def save_metrics(
            self,
            epoch,
            node2vec_loss,
            loss_train,
            loss_val,
            loss_test,
            auc_train,
            auc_val,
            auc_test):

        metrics_path = self.model_metrics_path.format(
            p=self.p,
            q=self.q,
            epoch=epoch)
        metrics_folder = metrics_path.rsplit('/', 1)[0]
        if not os.path.exists(metrics_folder):
            os.makedirs(metrics_folder)
        if not os.path.exists(metrics_path):
            header = ','.join(METRICS_COLS) + '\n'
            with open(metrics_path, 'w') as stream:
                stream.write(header)

        metrics = [
            self.p,
            self.q,
            self.run,
            epoch,
            node2vec_loss,
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
            neg_edges_train,
            neg_edges_val,
            neg_edges_test,
            edges_train_auc,
            edges_val_auc,
            edges_test_auc,
            y):

        loss_train, loss_val, loss_test, auc_train, auc_val, auc_test = self.eval(
            edges_train,
            edges_val,
            edges_test,
            neg_edges_train,
            neg_edges_val,
            neg_edges_test,
            edges_train_auc,
            edges_val_auc,
            edges_test_auc,
            y)
        self.save_metrics(
            epoch=0,
            node2vec_loss=None,
            loss_train=loss_train.item(),
            loss_val=loss_val.item(),
            loss_test=loss_test.item(),
            auc_train=auc_train,
            auc_val=auc_val,
            auc_test=auc_test)

        for epoch in range(1, 1 + self.epochs):
            node2vec_loss = self.train_epoch()

            if epoch % self.eval_steps == 0:
                loss_train, loss_val, loss_test, auc_train, auc_val, auc_test = self.eval(
                    edges_train,
                    edges_val,
                    edges_test,
                    neg_edges_train,
                    neg_edges_val,
                    neg_edges_test,
                    edges_train_auc,
                    edges_val_auc,
                    edges_test_auc,
                    y)

                logging.info(
                    f'# P: {self.p}, '
                    f'# Q: {self.q}, '
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
                    epoch=epoch,
                    node2vec_loss=node2vec_loss.item(),
                    loss_train=loss_train.item(),
                    loss_val=loss_val.item(),
                    loss_test=loss_test.item(),
                    auc_train=auc_train,
                    auc_val=auc_val,
                    auc_test=auc_test)

    @classmethod
    def read_metrics(self, dataset, p, q):
        metrics_path = METRICS_PATH.format(
            dataset=dataset,
            p=p,
            q=q)
        return pd.read_csv(metrics_path)

    def discard_run_not_optimal_models(self):
        not_optimal_df = self.read_metrics(self.p, self.q)\
            .query(f'run == {self.run} & epoch != 0')\
            .sort_values('auc_val')\
            .iloc[:-1]

        for _, row in not_optimal_df.iterrows():
            predictor_path = self.predictor_path_pat.format(
                run=row['run'],
                p=row['p'],
                q=row['q'],
                epoch=row['epoch'])
            embedding_path = self.embedding_path_pat.format(
                run=row['run'],
                p=row['p'],
                q=row['q'],
                epoch=row['epoch'])

            try:
                logging.info('Deleting %s', predictor_path)
                os.remove(predictor_path)
            except:
                logging.info('%s does not exists', predictor_path)

            try:
                logging.info('Deleting %s', embedding_path)
                os.remove(embedding_path)
            except:
                logging.info('%s does not exists', embedding_path)


    @classmethod
    def load_model(
        self,
        run,
        edge_index,
        num_nodes,
        p,
        q,
        epoch,
        device,
        eval_steps=1,
        epochs=100,
        batch_size=128 * 1024
    ):

        omega = PositionalOmegaNode2Vec(
            device,
            num_nodes,
            edge_index,
            num_nodes,
            p,
            q,
            run=run,
            eval_steps=eval_steps,
            epochs=epochs,
            batch_size=batch_size)

        model_path = omega.model_path_pat.format(
            run=run,
            p=p,
            q=q,
            epoch=epoch)

        model = Node2Vec(
            edge_index,
            EMBEDDING_DIM,
            WALK_LENGTH,
            CONTEXT_SIZE,
            walks_per_node=WALKS_PER_NODE,
            p=self.p,
            q=self.q,
            num_negative_samples=WALKS_PER_NODE,
            num_nodes=num_nodes).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        predictor_path = omega.predictor_path_pat.format(
            run=omega.run,
            p=omega.p,
            q=omega.q,
            epoch=epoch)

        predictor = LinkPredictor().to(device)
        predictor.load_state_dict(torch.load(predictor_path))
        predictor.eval()

        return omega
