import logging
import os
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader

from src.torch_geo_models import GCN, LinkPredictor

HIDDEN_CHANNELS = 128
IN_CHANNELS = 128
DROPOUT = 0.5
LEARNING_RATIO = 0.005
COSSIM_MODEL_PATH_PAT = "models/structural_omega_gcn_cossim/{dataset}/model_{run}run_{n_layers_gcn}gslayers_epoch{epoch:04d}.pt"
COSSIM_PREDICTOR_PATH_PAT = "models/structural_omega_gcn_cossim/{dataset}/link_predictor_{run}run_{n_layers_gcn}gslayers_epoch{epoch:04d}.pt"
COSSIM_METRICS_PATH = (
    "data/metrics/{dataset}structural_omega_gcn_cossim_{n_layers_gcn}gslayers.csv"
)
METRICS_COLS = [
    "run",
    "epoch",
    "loss_train",
    "loss_val",
    "loss_test",
    "auc_train",
    "auc_val",
    "auc_test",
]


class StructuralOmegaGCNCosSim:
    def __init__(
        self,
        device,
        dataset,
        in_channels=IN_CHANNELS,
        eval_steps=50,
        n_layers_gcn=1,
        epochs=2000,
        batch_size=128 * 1024,
        run=0,
    ):
        self.n_layers_gcn = n_layers_gcn
        self.dataset = dataset
        self.in_channels = in_channels
        self.initialize_models_data(device)
        self.eval_steps = eval_steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.predictor_path_pat = COSSIM_PREDICTOR_PATH_PAT
        self.model_path_pat = COSSIM_MODEL_PATH_PAT
        self.model_metrics_path = COSSIM_METRICS_PATH
        self.run = run
        self.loss = torch.nn.BCELoss(reduction="mean")

    def initialize_models_data(self, device):
        self.model = GCN(
            self.n_layers_gcn,
            self.in_channels,
            HIDDEN_CHANNELS,
            HIDDEN_CHANNELS,
            DROPOUT,
        ).to(device)
        self.model.reset_parameters()

        self.predictor = LinkPredictor().to(device)
        self.predictor.reset_parameters()

        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.predictor.parameters()),
            lr=LEARNING_RATIO,
        )

    def train_epoch(self, edges_train, batch_size, x, y, adj_t):

        self.predictor.train()

        total_loss = total_examples = 0
        for perm in DataLoader(edges_train, batch_size, shuffle=True):

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
        for batch in DataLoader(edges, self.batch_size, shuffle=False):

            h = self.model(x, adj_t)
            batch_probas = self.predictor(h[batch[0]], h[batch[1]])
            probas.append(batch_probas)
        return torch.concat(probas)

    def is_same_class(self, edges, y):
        return (y[edges[0]] == y[edges[1]]) * 1

    def eval(self, edges_train, edges_val, edges_test, x, y, adj_t):
        self.predictor.eval()

        losses = []
        aucs = []

        # Loss evaluation
        for edges in [edges_train, edges_val, edges_test]:
            proba = self.forward(edges, x, adj_t)
            theta = self.is_same_class(edges, y).to(torch.float32)

            loss = self.loss(proba, theta)
            losses.append(loss.item())

            proba = proba.to("cpu").detach().numpy()
            theta = theta.to("cpu").detach().numpy()
            auc = roc_auc_score(theta, proba)
            aucs.append(auc)

        return *losses, *aucs

    def save_models(self, epoch):

        model_path = self.model_path_pat.format(
            dataset=self.dataset,
            run=self.run,
            n_layers_gcn=self.n_layers_gcn,
            epoch=epoch
        )
        model_folder = model_path.rsplit("/", 1)[0]
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        predictor_path = self.predictor_path_pat.format(
            dataset=self.dataset,
            run=self.run,
            n_layers_gcn=self.n_layers_gcn,
            epoch=epoch
        )
        predictor_folder = predictor_path.rsplit("/", 1)[0]
        if not os.path.exists(predictor_folder):
            os.makedirs(predictor_folder)

        torch.save(self.model.state_dict(), model_path)
        torch.save(self.predictor.state_dict(), predictor_path)

    def save_metrics(
        self, epoch, loss_train, loss_val, loss_test, auc_train, auc_val, auc_test
    ):

        metrics_path = self.model_metrics_path.format(
            dataset=self.dataset,
            n_layers_gcn=self.n_layers_gcn,
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
        ]

        metrics_row = ",".join(str(x) for x in metrics) + "\n"
        with open(metrics_path, "a") as stream:
            stream.write(metrics_row)

    def train(self, edges_train, edges_val, edges_test, x, y, adj_t):

        loss_train, loss_val, loss_test, auc_train, auc_val, auc_test = self.eval(
            edges_train, edges_val, edges_test, x, y, adj_t
        )
        self.save_metrics(
            0, loss_train, loss_val, loss_test, auc_train, auc_val, auc_test
        )

        for epoch in range(1, 1 + self.epochs):
            self.train_epoch(edges_train, self.batch_size, x, y, adj_t)

            if epoch % self.eval_steps == 0:
                (
                    loss_train,
                    loss_val,
                    loss_test,
                    auc_train,
                    auc_val,
                    auc_test,
                ) = self.eval(edges_train, edges_val, edges_test, x, y, adj_t)

                logging.info(
                    f"# GCN Layers: {self.n_layers_gcn}, "
                    f"Run: {self.run:04d}, "
                    f"Epoch: {epoch:04d}, "
                    f"Train Loss: {loss_train:.4f}, "
                    f"Valid loss: {loss_val:.4f}, "
                    f"Test loss: {loss_test:.4f}, "
                    f"Train AUC: {auc_train:.4f}, "
                    f"Valid AUC: {auc_val:.4f}, "
                    f"Test AUC: {auc_test:.4f}"
                )

                self.save_models(epoch)

                self.save_metrics(
                    epoch, loss_train, loss_val, loss_test, auc_train, auc_val, auc_test
                )

    @staticmethod
    def read_metrics(dataset, n_layers_gcn=1, n_layers_mlp=1):
        metrics_path = COSSIM_METRICS_PATH.format(
            dataset=dataset,
            n_layers_gcn=n_layers_gcn
        )
        return pd.read_csv(metrics_path)

    @classmethod
    def load_model(
        cls,
        dataset,
        run,
        epoch,
        device,
        eval_steps=50,
        n_layers_gcn=1,
        epochs=2000,
        batch_size=128 * 1024,
    ):

        omega = cls(
            device,
            run=run,
            eval_steps=eval_steps,
            n_layers_gcn=n_layers_gcn,
            epochs=epochs,
            batch_size=batch_size,
        )

        model_path = omega.model_path_pat.format(
            dataset=dataset,
            run=omega.run,
            n_layers_gcn=omega.n_layers_gcn,
            epoch=epoch
        )

        model = GCN(
            omega.n_layers_gcn,
            IN_CHANNELS,
            HIDDEN_CHANNELS,
            HIDDEN_CHANNELS,
            DROPOUT
        ).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        predictor_path = omega.predictor_path_pat.format(
            dataset=dataset,
            run=omega.run,
            n_layers_gcn=omega.n_layers_gcn,
            epoch=epoch
        )

        predictor = LinkPredictor().to(device)
        predictor.load_state_dict(torch.load(predictor_path))
        predictor.eval()

        omega.model = model
        omega.predictor = predictor

        return omega
