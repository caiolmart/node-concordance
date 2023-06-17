import logging
import os
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from torch_geometric.loader import DataLoader

from src.torch_geo_models import GraphSAGE, LinkPredictor


HIDDEN_CHANNELS = 128
IN_CHANNELS = 128
OUT_CHANNELS = 40
DROPOUT = 0.5
LEARNING_RATIO = 0.005
# DOTMLP_MODEL_PATH_PAT = "models/node_classifier_grapn_sage_dotmlp/{dataset}/model_{run}run_{n_layers_graph_sage}gslayers_{n_layers_mlp}mlplayers_epoch{epoch:04d}.pt"
# DOTMLP_PREDICTOR_PATH_PAT = "models/node_classifier_grapn_sage_dotmlp/{dataset}/link_predictor_{run}run_{n_layers_graph_sage}gslayers_{n_layers_mlp}mlplayers_epoch{epoch:04d}.pt"
# DOTMLP_METRICS_PATH = "data/metrics/{dataset}/node_classifier_grapn_sage_dotmlp_{n_layers_graph_sage}gslayers_{n_layers_mlp}mlplayers.csv"
COSSIM_MODEL_PATH_PAT = "models/node_classifier_graph_sage_cossim/{dataset}/model_{run}run_{n_layers_graph_sage}gslayers_epoch{epoch:04d}.pt"
COSSIM_PREDICTOR_PATH_PAT = "models/node_classifier_graph_sage_cossim/{dataset}/link_predictor_{run}run_{n_layers_graph_sage}gslayers_epoch{epoch:04d}.pt"
COSSIM_METRICS_PATH = "data/metrics/{dataset}/node_classifier_graph_sage_cossim_{n_layers_graph_sage}gslayers.csv"
METRICS_COLS = [
    "run",
    "epoch",
    "loss_train",
    "loss_val",
    "loss_test",
    "auc_train",
    "auc_val",
    "auc_test",
    "nodes_acc_train",
    "nodes_acc_val",
    "nodes_acc_test",
]


class NodeClassifierOmegaGraphSageCosSim:
    def __init__(
        self,
        device,
        dataset,
        node_evaluator,
        in_channels=IN_CHANNELS,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=OUT_CHANNELS,
        eval_steps=100,
        n_layers_graph_sage=1,
        epochs=5000,
        batch_size=1024 * 1024,
        run=0,
    ):
        self.dataset = dataset
        self.n_layers_graph_sage = n_layers_graph_sage
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.initialize_models_data(device)
        self.eval_steps = eval_steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.predictor_path_pat = COSSIM_PREDICTOR_PATH_PAT
        self.model_path_pat = COSSIM_MODEL_PATH_PAT
        self.model_metrics_path = COSSIM_METRICS_PATH
        self.run = run
        self.loss = torch.nn.NLLLoss(reduction="mean")
        self.node_evaluator = node_evaluator

    def initialize_models_data(self, device):
        self.model = GraphSAGE(
            self.n_layers_graph_sage,
            self.in_channels,
            self.hidden_channels,
            self.out_channels,
            DROPOUT,
        ).to(device)
        self.model.reset_parameters()

        self.predictor = LinkPredictor().to(device)
        self.predictor.reset_parameters()

        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()), lr=LEARNING_RATIO
        )

    def train_epoch(self, nodes_train, x, y, adj_t):
        self.model.train()
        self.optimizer.zero_grad()

        h = self.model(x, adj_t)[nodes_train]
        predicted = h.log_softmax(dim=-1)
        y_true = y[nodes_train].squeeze(1)
        loss = self.loss(predicted, y_true)
        loss.backward()

        self.optimizer.step()

        return loss

    def forward(self, edges, x, adj_t):
        probas = []
        for batch in DataLoader(edges, self.batch_size, shuffle=False):
            h = self.model(x, adj_t)
            predicted = h.log_softmax(dim=-1)
            batch_probas = self.predictor(
                predicted[batch[0]], predicted[batch[1]]
            )
            probas.append(batch_probas)
        return torch.concat(probas)

    def is_same_class(self, edges, y):
        return (y[edges[0]] == y[edges[1]]) * 1

    def eval(
        self,
        edges_train,
        edges_val,
        edges_test,
        nodes_train,
        nodes_val,
        nodes_test,
        x,
        y,
        adj_t,
    ):
        self.model.eval()

        losses = []
        aucs = []
        node_accs = []

        # Loss evaluation
        for edges in [edges_train, edges_val, edges_test]:
            proba = self.forward(edges, x, adj_t)
            theta = self.is_same_class(edges, y).to(torch.float32)

            proba = proba.to("cpu").detach().numpy()
            theta = theta.to("cpu").detach().numpy()
            auc = roc_auc_score(theta, proba)
            aucs.append(auc)

        h = self.model(x, adj_t)
        predicted = h.log_softmax(dim=-1)
        y_pred = predicted.argmax(dim=-1, keepdim=True)
        for nodes in [nodes_train, nodes_val, nodes_test]:
            loss = self.loss(predicted[nodes], y[nodes].squeeze(1))
            losses.append(loss.item())

            acc = self.node_evaluator.eval(
                {
                    "y_true": y[nodes],
                    "y_pred": y_pred[nodes],
                }
            )["acc"]

            node_accs.append(acc)

        return *losses, *aucs, *node_accs

    def save_models(self, epoch):
        model_path = self.model_path_pat.format(
            dataset=self.dataset,
            run=self.run,
            n_layers_graph_sage=self.n_layers_graph_sage,
            epoch=epoch,
        )
        model_folder = model_path.rsplit("/", 1)[0]
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        predictor_path = self.predictor_path_pat.format(
            dataset=self.dataset,
            run=self.run,
            n_layers_graph_sage=self.n_layers_graph_sage,
            epoch=epoch,
        )
        predictor_folder = predictor_path.rsplit("/", 1)[0]
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
        auc_test,
        nodes_acc_train,
        nodes_acc_val,
        nodes_acc_test,
    ):
        metrics_path = self.model_metrics_path.format(
            dataset=self.dataset,
            n_layers_graph_sage=self.n_layers_graph_sage,
        )
        metrics_folder = metrics_path.rsplit("/", 1)[0]
        if not os.path.exists(metrics_folder):
            os.makedirs(metrics_folder)
        if not os.path.exists(metrics_path):
            header = ",".join(METRICS_COLS) + "\n"
            with open(metrics_path, "w") as stream:
                stream.write(header)

        metrics = [
            self.run,
            epoch,
            loss_train,
            loss_val,
            loss_test,
            auc_train,
            auc_val,
            auc_test,
            nodes_acc_train,
            nodes_acc_val,
            nodes_acc_test,
        ]

        metrics_row = ",".join(str(x) for x in metrics) + "\n"
        with open(metrics_path, "a") as stream:
            stream.write(metrics_row)

    def train(
        self,
        edges_train,
        edges_val,
        edges_test,
        nodes_train,
        nodes_val,
        nodes_test,
        x,
        y,
        adj_t,
    ):
        (
            loss_train,
            loss_val,
            loss_test,
            auc_train,
            auc_val,
            auc_test,
            nodes_acc_train,
            nodes_acc_val,
            nodes_acc_test,
        ) = self.eval(
            edges_train,
            edges_val,
            edges_test,
            nodes_train,
            nodes_val,
            nodes_test,
            x,
            y,
            adj_t,
        )
        self.save_metrics(
            0,
            loss_train,
            loss_val,
            loss_test,
            auc_train,
            auc_val,
            auc_test,
            nodes_acc_train,
            nodes_acc_val,
            nodes_acc_test,
        )

        for epoch in range(1, 1 + self.epochs):
            self.train_epoch(nodes_train, x, y, adj_t)

            if epoch % self.eval_steps == 0:
                (
                    loss_train,
                    loss_val,
                    loss_test,
                    auc_train,
                    auc_val,
                    auc_test,
                    nodes_acc_train,
                    nodes_acc_val,
                    nodes_acc_test,
                ) = self.eval(
                    edges_train,
                    edges_val,
                    edges_test,
                    nodes_train,
                    nodes_val,
                    nodes_test,
                    x,
                    y,
                    adj_t,
                )

                logging.info(
                    f"# Graph Sage Layers: {self.n_layers_graph_sage}, "
                    f"Run: {self.run:04d}, "
                    f"Epoch: {epoch:04d}, "
                    f"Train Loss: {loss_train:.4f}, "
                    f"Valid loss: {loss_val:.4f}, "
                    f"Test loss: {loss_test:.4f}, "
                    f"Train AUC: {auc_train:.4f}, "
                    f"Valid AUC: {auc_val:.4f}, "
                    f"Test AUC: {auc_test:.4f}, "
                    f"Node Train ACC: {nodes_acc_train:.4f}, "
                    f"Node Valid ACC: {nodes_acc_val:.4f}, "
                    f"Node Test ACC: {nodes_acc_test:.4f}, "
                )

                self.save_models(epoch)

                self.save_metrics(
                    epoch,
                    loss_train,
                    loss_val,
                    loss_test,
                    auc_train,
                    auc_val,
                    auc_test,
                    nodes_acc_train,
                    nodes_acc_val,
                    nodes_acc_test,
                )

    @staticmethod
    def read_metrics(dataset, n_layers_graph_sage=1, n_layers_mlp=1):
        metrics_path = COSSIM_METRICS_PATH.format(
            dataset=dataset, n_layers_graph_sage=n_layers_graph_sage
        )
        return pd.read_csv(metrics_path)

    @classmethod
    def load_model(
        cls,
        dataset,
        run,
        epoch,
        device,
        eval_steps=100,
        n_layers_graph_sage=1,
        epochs=5000,
        batch_size=128 * 1024,
    ):
        omega = cls(
            device,
            dataset=dataset,
            run=run,
            eval_steps=eval_steps,
            n_layers_graph_sage=n_layers_graph_sage,
            epochs=epochs,
            batch_size=batch_size,
        )

        model_path = omega.model_path_pat.format(
            dataset=omega.dataset,
            run=omega.run,
            n_layers_graph_sage=omega.n_layers_graph_sage,
            epoch=epoch,
        )

        model = GraphSAGE(
            omega.n_layers_graph_sage, IN_CHANNELS, HIDDEN_CHANNELS, DROPOUT
        ).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        predictor_path = omega.predictor_path_pat.format(
            dataset=omega.dataset,
            run=omega.run,
            n_layers_graph_sage=omega.n_layers_graph_sage,
            epoch=epoch,
        )

        predictor = LinkPredictor().to(device)
        predictor.load_state_dict(torch.load(predictor_path))
        predictor.eval()

        omega.model = model
        omega.predictor = predictor

        return omega
