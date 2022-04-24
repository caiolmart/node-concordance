import torch
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
from torch_geometric.data.data import Data
from torch_geometric.utils import negative_sampling


EDGES_SEED = 12345
TEST_VAL_PROPORTION = 0.025
DATA_FOLDER = 'data/gamma/'
DATA_NAME = 'ogbn-arxiv'


def remove_rows(tensor, indices):
    mask = torch.ones(tensor.size(1), dtype=torch.bool)
    mask[indices] = False
    return tensor[:, mask]


def load_data() -> Data:
    dataset = PygNodePropPredDataset(name=DATA_NAME, 
                                     root=DATA_FOLDER)
    data = dataset[0]
    return data


def get_val_test_edges(data: Data, remove_from_data=True, device=None):
    # Getting test and validation edges
    torch.manual_seed(EDGES_SEED)
    idx = torch.randperm(data.edge_index.size(1))
    n_val_edges = n_test_edges =\
        int(data.edge_index.size(1) * TEST_VAL_PROPORTION)
    val_idx = idx[:n_val_edges]
    test_idx = idx[n_val_edges:(n_val_edges + n_test_edges)]
    edges_val = data.edge_index[:, val_idx]
    edges_test = data.edge_index[:, test_idx]

    neg_edges_val = negative_sampling(
        data.edge_index,
        num_nodes=data.x.size(0),
        num_neg_samples=edges_val.size(1),
        method='sparse')
    neg_edges_test = negative_sampling(
        data.edge_index,
        num_nodes=data.x.size(0),
        num_neg_samples=edges_test.size(1),
        method='sparse')

    # Removing validation and test edges
    if remove_from_data:
        remove_idx = torch.cat([val_idx, test_idx])
        data.edge_index = remove_rows(data.edge_index, remove_idx)

    if device:
        data = data.to(device)
        edges_val = edges_val.to(device)
        edges_test = edges_test.to(device)

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
