from typing import Tuple
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.data.data import Data


DATA_FOLDER = "data/gamma/torch_cora"
NAME='Cora'
SPLIT='full'


def load_dataset() -> Data:
    dataset = Planetoid(root=DATA_FOLDER,
                        name=NAME,
                        split=SPLIT)
    return dataset


def get_train_val_test_edges(
        dataset: Planetoid,
        remove_from_data=False,
        device=None
    ) -> Tuple[Data, torch.tensor, torch.tensor, torch.tensor]:

    data = dataset[0]

    # Creating node -> flag map (train, valid, test)
    nodes = torch.arange(data.num_nodes)
    idx_split_map = {}
    train_mask = data.train_mask
    for idx in nodes[train_mask].numpy():
        idx_split_map[idx] = 0
    val_mask = data.val_mask
    for idx in nodes[val_mask].numpy():
        idx_split_map[idx] = 1
    test_mask = data.test_mask
    for idx in nodes[test_mask].numpy():
        idx_split_map[idx] = 2

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
