import torch
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
from torch_geometric.data.data import Data


EDGES_SEED = 12345
TEST_VAL_PROPORTION = 0.025
DATA_FOLDER = 'data/gamma/'
DATA_NAME = 'ogbn-arxiv'


def remove_rows(tensor, indices):
    mask = torch.ones(tensor.size(1), dtype=torch.bool)
    mask[indices] = False
    return tensor[:, mask]


def load_data():
    dataset = PygNodePropPredDataset(name=DATA_NAME, 
                                     root=DATA_FOLDER)
    data = dataset[0]

    # Getting test and validation edges
    torch.manual_seed(EDGES_SEED)
    idx = torch.randperm(data.edge_index.size(1))
    n_val_edges = n_test_edges = int(data.edge_index.size(1) * TEST_VAL_PROPORTION)
    val_idx = idx[:n_val_edges]
    test_idx = idx[n_val_edges:(n_val_edges + n_test_edges)]
    val_edges = data.edge_index[:, val_idx]
    test_edges = data.edge_index[:, test_idx]

    # Removing validation and test edges
    remove_idx = torch.cat([val_idx, test_idx])
    data.edge_index = remove_rows(data.edge_index, remove_idx)

    return data, val_edges, test_edges


def prepare_adjencency(data: Data):
    T.ToSparseTensor()(data)
    data.adj_t = data.adj_t.to_symmetric()

    return data