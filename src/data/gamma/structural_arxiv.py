from typing import Tuple
import torch
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
from torch_geometric.data.data import Data


DATA_FOLDER = 'data/gamma/'
DATA_NAME = 'ogbn-arxiv'


def load_dataset() -> Data:
    dataset = PygNodePropPredDataset(name=DATA_NAME,
                                     root=DATA_FOLDER)
    return dataset


def get_train_val_test_edges(
        dataset: PygNodePropPredDataset,
        remove_from_data=False,
        device=None
    ) -> Tuple[Data, torch.tensor, torch.tensor, torch.tensor]:

    # Creating node -> flag map (train, valid, test)
    idx_split = dataset.get_idx_split()
    idx_split_map = {}
    for idx in idx_split['train'].numpy():
        idx_split_map[idx] = 0
    for idx in idx_split['valid'].numpy():
        idx_split_map[idx] = 1
    for idx in idx_split['test'].numpy():
        idx_split_map[idx] = 2

    data = dataset[0]

    # Creating edge flags
    node_flags = data.edge_index.clone()
    node_flags.apply_(lambda val: idx_split_map.get(val))
    edge_flags = torch.max(node_flags, axis=0).values

    # Getting train, validation and test edges
    train_mask = edge_flags == 0
    edges_train = data.edge_index[:, train_mask]

    val_mask = edge_flags == 1
    edges_val = data.edge_index[:, val_mask]

    test_mask = edge_flags == 2
    edges_test = data.edge_index[:, test_mask]

    # Removing validation and test edges
    if remove_from_data:
        data.edge_index = data.edge_index[:, train_mask]

    if device:
        data = data.to(device)
        edges_train = edges_train.to(device)
        edges_val = edges_val.to(device)
        edges_test = edges_test.to(device)

    return data, edges_train, edges_val, edges_test


def prepare_adjencency(data: Data, to_symmetric=True):
    T.ToSparseTensor()(data)
    if to_symmetric:
        data.adj_t = data.adj_t.to_symmetric()

    return data


def get_edge_index_from_adjencency(data: Data, device=None):
    row, col, _ = data.adj_t.coo()
    edge_index = torch.stack([col, row], dim=0).to(device)
    return edge_index
