import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data.data import Data
from ogb.nodeproppred import DglNodePropPredDataset, PygNodePropPredDataset
from dgl.heterograph import DGLHeteroGraph
from torch_sparse import SparseTensor


EDGES_SEED = 12345
TEST_VAL_PROPORTION = 0.025
DATA_FOLDER_DGL = 'data/node_classifier/'
DATA_NAME_DGL = 'ogbn-arxiv'
DATA_FOLDER_PYG = 'data/gamma/'
DATA_NAME_PYG = 'ogbn-arxiv'


def remove_rows(tensor, indices):
    mask = torch.ones(tensor.size(1), dtype=torch.bool)
    mask[indices] = False
    return tensor[:, mask]


def load_dataset_dgl():
    dataset = DglNodePropPredDataset(name=DATA_NAME_DGL, 
                                     root=DATA_FOLDER_DGL)
    return dataset


def get_symmetric_graph_dgl(dataset) -> DGLHeteroGraph:
    graph = dataset[0][0]
    graph.add_edges(graph.edges()[1], graph.edges()[0])
    return graph


def load_dataset_pyg():
    dataset = PygNodePropPredDataset(name=DATA_NAME_PYG, 
                                     root=DATA_FOLDER_PYG)
    return dataset


def data_to_sparse_symmetric_pyg(data: Data) -> Data:
    T.ToSparseTensor()(data)
    data.adj_t = data.adj_t.to_symmetric()

    return data


def get_edge_weights_pyg(
        csv_path: str,
        n_nodes: int,
        device: torch.device,
        source_col='source',
        target_col='target',
        weight_col='gamma_pred'
    ) -> SparseTensor:
    edges_gamma_df = pd.read_csv(csv_path)
    edge_weights = SparseTensor.from_torch_sparse_coo_tensor(
        torch.sparse_coo_tensor(
            edges_gamma_df[[source_col, target_col]].values.T,
            edges_gamma_df[weight_col].values,
            size=(n_nodes, n_nodes)))\
        .to(device)
    return edge_weights
