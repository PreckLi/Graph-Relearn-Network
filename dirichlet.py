from torch_geometric.utils import k_hop_subgraph, degree
import torch
import numpy as np


def dirichlet_energy(x, data, k):
    dirichlet = 0
    for i in range(data.num_nodes):
        subsets, _, _, inv = k_hop_subgraph(i, k, data.edge_index)
        minus = x[subsets.tolist()]
        x_i = np.tile(x[i], (minus.shape[0], 1))
        dirichlet += np.linalg.norm(minus - x_i)**2
    dirichlet = dirichlet / (data.num_nodes)
    return dirichlet


def dirichlet_energy_regularizar(x, data, k):
    dirichlet = 0
    deg = degree(data.edge_index[0])
    for i in range(data.num_nodes):
        subsets, _, _, inv = k_hop_subgraph(i, k, data.edge_index)
        degree_subsets = np.sqrt(deg[subsets.tolist()].cpu().numpy() + 1)
        minus = np.divide(x[subsets.tolist()], degree_subsets.reshape(-1, 1))
        x_i = np.divide(np.tile(x[i], (minus.shape[0], 1)), np.sqrt(deg[i].cpu().numpy() + 1))
        dirichlet += np.sum(np.linalg.norm(minus - x_i))**2
    dirichlet = dirichlet / (data.num_nodes)
    return dirichlet
