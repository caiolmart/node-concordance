import pandas as pd
import torch
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.heterograph import DGLHeteroGraph


EDGES_SEED = 12345
TEST_VAL_PROPORTION = 0.025
DATA_FOLDER = 'data/node_classifier/'
DATA_NAME = 'ogbn-arxiv'


def remove_rows(tensor, indices):
    mask = torch.ones(tensor.size(1), dtype=torch.bool)
    mask[indices] = False
    return tensor[:, mask]


def load_dataset():
    dataset = DglNodePropPredDataset(name=DATA_NAME, 
                                     root=DATA_FOLDER)
    return dataset


def get_symmetric_graph(dataset) -> DGLHeteroGraph:
    graph = dataset[0][0]
    graph.add_edges(graph.edges()[1], graph.edges()[0])
    return graph
