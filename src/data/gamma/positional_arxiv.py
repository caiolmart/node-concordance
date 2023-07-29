from typing import Tuple
import numpy as np
import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data.data import Data
from torch_geometric.utils import negative_sampling

EDGES_SEED = 12345
TEST_VAL_PROPORTION = 0.01
DATA_FOLDER = "data/gamma/"
DATA_NAME = "ogbn-arxiv"


def remove_rows(tensor, indices):
    mask = torch.ones(tensor.size(1), dtype=torch.bool)
    mask[indices] = False
    return tensor[:, mask]


def load_dataset() -> Data:
    dataset = PygNodePropPredDataset(name=DATA_NAME, root=DATA_FOLDER)
    return dataset


def get_train_val_test_edges_auc(
    dataset: PygNodePropPredDataset, remove_from_data=False, device=None
) -> Tuple[Data, torch.tensor, torch.tensor, torch.tensor]:
    # Creating node -> flag map (train, valid, test)
    idx_split = dataset.get_idx_split()
    idx_split_map = {}
    for idx in idx_split["train"].numpy():
        idx_split_map[idx] = 0
    for idx in idx_split["valid"].numpy():
        idx_split_map[idx] = 1
    for idx in idx_split["test"].numpy():
        idx_split_map[idx] = 2

    data = dataset[0]

    # Creating edge flags
    node_flags = data.edge_index.clone()
    node_flags.apply_(lambda val: idx_split_map.get(val))
    edge_flags = torch.max(node_flags, axis=0).values

    # Getting train, validation and test edges
    train_mask = edge_flags == 0
    edges_train_auc = data.edge_index[:, train_mask]

    val_mask = edge_flags == 1
    edges_val_auc = data.edge_index[:, val_mask]

    test_mask = edge_flags == 2
    edges_test_auc = data.edge_index[:, test_mask]

    # Removing validation and test edges
    if remove_from_data:
        data.edge_index = data.edge_index[:, train_mask]

    if device:
        data = data.to(device)
        edges_train_auc = edges_train_auc.to(device)
        edges_val_auc = edges_val_auc.to(device)
        edges_test_auc = edges_test_auc.to(device)

    return data, edges_train_auc, edges_val_auc, edges_test_auc


def get_val_test_edges_link_pred(
    data: Data, remove_from_data=True, device=None, get_neg_edges_train=False
):
    # Getting test and validation edges
    torch.manual_seed(EDGES_SEED)
    idx = torch.randperm(data.edge_index.size(1))
    n_val_edges = n_test_edges = int(
        data.edge_index.size(1) * TEST_VAL_PROPORTION
    )
    val_idx = idx[:n_val_edges]
    test_idx = idx[n_val_edges : (n_val_edges + n_test_edges)]
    edges_val = data.edge_index[:, val_idx]
    edges_test = data.edge_index[:, test_idx]

    neg_edges_val = negative_sampling(
        data.edge_index,
        num_nodes=data.x.size(0),
        num_neg_samples=edges_val.size(1),
        method="sparse",
    )
    neg_edges_test = negative_sampling(
        data.edge_index,
        num_nodes=data.x.size(0),
        num_neg_samples=edges_test.size(1),
        method="sparse",
    )

    if get_neg_edges_train:
        neg_edges_train = negative_sampling(
            data.edge_index,
            num_nodes=data.x.size(0),
            num_neg_samples=edges_test.size(1),
            method="sparse",
        )

    # Removing validation and test edges
    if remove_from_data:
        remove_idx = torch.cat([val_idx, test_idx])
        data.edge_index = remove_rows(data.edge_index, remove_idx)

    if device:
        data = data.to(device)
        edges_val = edges_val.to(device)
        edges_test = edges_test.to(device)

    if get_neg_edges_train:
        return (
            data,
            edges_val,
            edges_test,
            neg_edges_val,
            neg_edges_test,
            neg_edges_train,
        )

    return data, edges_val, edges_test, neg_edges_val, neg_edges_test


def prepare_adjencency(data: Data, to_symmetric=True):
    T.ToSparseTensor()(data)
    if to_symmetric:
        data.adj_t = data.adj_t.to_symmetric()

    return data


def get_edge_index_from_adjencency(data: Data, device=None):
    row, col, _ = data.adj_t.coo()
    edge_index = torch.stack([col, row], dim=0).to(device)
    return edge_index


def get_train_val_test_edges_year(
    node_years: np.array,
    edge_index: torch.tensor,
    year: int,
    device=None,
) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    # Creating node -> flag map (train, valid, test)
    train_mask = node_years < year
    val_mask = node_years == year
    test_mask = node_years > year
    idx_split_array = np.zeros(node_years.shape[0])
    idx_split_array[train_mask] = 0
    idx_split_array[val_mask] = 1
    idx_split_array[test_mask] = 2
    idx_split_map = dict(zip(range(idx_split_array.shape[0]), idx_split_array))

    # Creating edge flags
    node_flags = edge_index.clone()
    node_flags.apply_(lambda val: idx_split_map.get(val))
    edge_flags = torch.max(node_flags, axis=0).values

    # Getting train, validation and test edges
    train_mask = edge_flags == 0
    edges_train = edge_index[:, train_mask]

    val_mask = edge_flags == 1
    edges_val = edge_index[:, val_mask]

    test_mask = edge_flags == 2
    edges_test = edge_index[:, test_mask]

    if device:
        edges_train = edges_train.to(device)
        edges_val = edges_val.to(device)
        edges_test = edges_test.to(device)

    return edges_train, edges_val, edges_test
