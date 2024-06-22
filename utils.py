from torch_geometric.datasets import Planetoid, Amazon, CitationFull, CoraFull, Coauthor, Airports,PolBlogs
from torch_geometric.utils import k_hop_subgraph, degree, to_networkx, to_dense_adj
from torch_geometric.nn.models import Node2Vec
from torch_geometric.data import Data
from models.GCN import GCN
from models.GATpyg import GAT
from models.GIN import GIN
from models.Chebnet import Chebnet
from models.GraphSAGE import GraphSAGE
from models.UniMP import UniMP
from models.GRN import GRN
from models.Arma import ARMA
from models.PAN import PAN
from models.LightGCN import LightGCN
from models.fusedgat import FusedGAT
from models.GPS import GPS
from madgap import mad_value
from dirichlet import dirichlet_energy,dirichlet_energy_regularizar
import scipy.sparse as sp
import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.cluster import KMeans, SpectralClustering
import random


def load_data(args):
    if args.dataset == 'terrorist':
        edge_index = get_edge_index("../datasets/TerroristRel/terroristrel.edgelist")
        label = get_label("../datasets/TerroristRel/label.txt")
        # x = torch.eye(label.shape[0])
        x = torch.ones((label.shape[0],label.shape[0]))
        data = Data(x=x, edge_index=edge_index, num_nodes=x.shape[0], num_classes=2, y=label)
        if args.split == 'random':
            data = random_split(args, data)
        if args.split == 'default':
            data = default_split(args, data)
        return data
    if args.dataset == "cora":
        dataset = Planetoid(root='../datasets/Planetoid', name='Cora')
    if args.dataset == "citeseer":
        dataset = Planetoid(root="../datasets/Planetoid", name="Citeseer")
    if args.dataset == "pubmed":
        dataset = Planetoid(root="../datasets/Planetoid", name="Pubmed")
    if args.dataset == "photo":
        dataset = Amazon(root="../datasets", name="Photo")
    if args.dataset == "dblp":
        dataset = CitationFull(root="../datasets", name="DBLP")
    if args.dataset == "physics":
        dataset = Coauthor(root="../datasets", name="Physics")
    if args.dataset == "cs":
        dataset = Coauthor(root="../datasets", name="CS")
    if args.dataset == "corafull":
        dataset = CoraFull(root="../datasets/Corafull")
    if args.dataset == "brazil":
        dataset = Airports(root="../datasets", name="Brazil")
    if args.dataset == "usa":
        dataset = Airports(root="../datasets", name="USA")
    if args.dataset == "europe":
        dataset = Airports(root="../datasets", name="Europe")
    if args.dataset == 'computers':
        dataset = Amazon(root='../datasets',name='Computers')
    if args.dataset =='polblogs':
        dataset = PolBlogs(root='../datasets')

    data = dataset[0]
    data.num_classes = dataset.num_classes
    # data.x = normalize(data.x)
    if args.split == 'random':
        data = random_split(args, data)
    if args.split == 'default':
        data = default_split(args, data)
    return data

def random_split(args, data):
    train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)

    train_idx = np.random.choice(data.num_nodes, int(data.num_nodes * args.train_ratio), replace=False)
    residue = np.array(list(set(range(data.num_nodes)) - set(train_idx)))
    val_idx = np.random.choice(residue, int(data.num_nodes * args.val_ratio), replace=False)
    test_idx = np.array(list(set(residue) - set(val_idx)))

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data


def default_split(args, data):
    train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    train_mask[:int(args.train_ratio * data.y.size(0))] = True

    val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    val_mask[int((args.train_ratio) * data.y.size(0)):int((args.train_ratio + args.val_ratio) * data.y.size(0))] = True

    test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    test_mask[int((args.train_ratio + args.val_ratio) * data.y.size(0)):] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data


def load_model(args, data):
    if args.gnn_style == 'gcn':
        model = GCN(data, args)
    if args.gnn_style == 'gat':
        model = GAT(data.num_node_features, args.hid_size, data.num_classes, args.dropout, 0.2, args.gat_heads)
    if args.gnn_style[:3] == 'gin':
        model = GIN(data, args)
    if args.gnn_style == 'graphsage':
        model = GraphSAGE(data, args)
    if args.gnn_style == 'chebnet':
        model = Chebnet(data, args)
    if args.gnn_style == 'unimp':
        model = UniMP(data, args)
    if args.gnn_style == 'arma':
        model = ARMA(data, args)
    if args.gnn_style == 'pan':
        model = PAN(data, args)
    if args.gnn_style == 'lgcn':
        model = LightGCN(data, args)
    if args.gnn_style[:3] == 'grn':
        model = GRN(data, args)
    if args.gnn_style == 'fusedgat':
        model = FusedGAT(data, args)
    if args.gnn_style == 'gps':
        model = GPS(data, args)
    return model


def dense_to_pyg_edge_index(adj):
    adj = sp.coo_matrix(adj)
    indices = np.vstack((adj.row, adj.col))
    adj = torch.LongTensor(indices)
    return adj


def plot_subset(node_idx, subset, subset_edge_index, label, whole_label, k):
    G = nx.Graph()
    G.add_nodes_from(subset)
    for i, j in zip(subset_edge_index[0], subset_edge_index[1]):
        G.add_edge(i, j)
    color_dict = {0: "r", 1: "g", 2: "b", 3: "c", 4: "m", 5: "y", 6: "darkorange", 7: "moccasin"}
    color_nodes = [color_dict[idx] for idx in label]
    plt.figure()
    plt.title(f"Epoch:{k}, Central Node:{node_idx}, label:{color_dict[int(whole_label[node_idx])]}")
    nx.draw(G, node_size=400, font_size=12, with_labels=True, pos=nx.spring_layout(G, k=0.4), node_color=color_nodes)
    plt.tight_layout()
    plt.show()
    pass


def get_all_neighs(num_nodes, edge_index):
    neighs_list = list()
    for i in range(num_nodes):
        subset, _, _, _ = k_hop_subgraph(i, 1, edge_index)
        neighs_list.append(subset)
    return neighs_list


def get_neighs(central_list, edge_index, args):
    neighs_list = list()
    for i in central_list:
        subset, _, _, _ = k_hop_subgraph(int(i), args.khops, edge_index)
        neighs_list.append(subset)
    subsets, new_edge_index, invs, _ = k_hop_subgraph(central_list, args.khops, edge_index, relabel_nodes=True)
    return neighs_list, subsets, new_edge_index, invs

def get_changable_nodes(pred_array, args, m):
    changable_nodes = []
    for i in range(len(pred_array)):
        for epoch in range(int(m + 1.0) * args.up - args.area, int(m + 1.0) * args.up):
            if pred_array[i][epoch] == pred_array[i][int(m + 1.0) * args.up - args.area]:
                continue
            else:
                break
        if epoch < int(m + 1.0) * args.up - 1:
            changable_nodes.append(i)
    return changable_nodes


def get_changable_nodes2(pred_array, args, m):
    changable_array = list()
    for i in range(len(pred_array)):
        temp_list = list()
        for epoch in range(int((m + 1.0) * args.up - args.area), int((m + 1.0) * args.up)):
            if pred_array[i][epoch - 1] == pred_array[i][epoch]:
                temp_list.append(0)
            else:
                temp_list.append(1)
        changable_array.append(temp_list)
    # model = KMeans(n_clusters=2)
    model = SpectralClustering(n_clusters=2)
    model.fit(changable_array)
    # result = model.predict(changable_array)
    result = model.labels_
    if len(result[result == 0]) > len(result[result == 1]):
        changable_nodes = list(np.where(result == 1)[0])
    else:
        changable_nodes = list(np.where(result == 0)[0])
    return changable_nodes

def get_edge_index(file):
    edge_list_file = open(file, encoding="utf-8")
    edge_list = list()
    for reader in edge_list_file.readlines():
        row = str(reader).strip("\n").split(" ")
        temprow = list()
        for i in row:
            temprow.append(int(i))
        edge_list.append(temprow)
    edge_list = sorted(sorted(edge_list, key=(lambda x: x[1])), key=(lambda x: x[0]))
    edge_list = list(map(list, zip(*edge_list)))
    edge_list = torch.tensor(edge_list, dtype=torch.long)
    return edge_list

def get_label(file):
    node_labels_list = list()
    label_list = list()
    cnt = 0
    file = open(file, encoding="utf-8")
    for reader in file.readlines():
        if cnt == 0:
            cnt = 1
            continue
        row = str(reader).strip("\n").split(" ")
        temprow = list()
        for i in row:
            temprow.append(int(i))
        node_labels_list.append(temprow)
        cnt += 1
    for i in node_labels_list:
        label_list.append(i[1])
    label = torch.as_tensor(label_list, dtype=torch.long)
    return label

def del_tensor_ele(arr, index):
    arr1 = arr[0:index]
    arr2 = arr[index + 1:]
    return torch.cat((arr1, arr2), dim=0)


def compute_len_degree(degrees):
    count = 0
    for i in degrees:
        if i != 0:
            count += 1
        else:
            break
    return count


def negative_sample(changable_nodes, edge_index, args):
    negative_nodes = list()
    if args.sample_style == 'degree':
        for i in changable_nodes:
            subset, sub_edge_index, _, _ = k_hop_subgraph(int(i), 1, edge_index)
            degrees = del_tensor_ele(degree(sub_edge_index[0]), i)
            degrees, indices = torch.sort(degrees, descending=True)
            len_degree = compute_len_degree(degrees)
            negative_nodes.extend(indices.tolist()[:int(args.sample_ratio * len_degree) + 1])
    if args.sample_style == 'random':
        for i in changable_nodes:
            subset, sub_edge_index, _, _ = k_hop_subgraph(int(i), 1, edge_index)
            subset_list = subset.tolist()
            negative_nodes.extend(random.sample(subset_list, int(args.sample_ratio * len(subset_list)) + 1))
    if args.sample_style == 'walk':
        for i in changable_nodes:
            subset, sub_edge_index, _, _ = k_hop_subgraph(int(i), 2, edge_index)
            sample_num = int(args.sample_ratio * subset.shape[0])
            graph = Data(edge_index=sub_edge_index)
            graph = to_networkx(graph)
            walk_list = k_hop_seq(graph, i, sample_num, [], [i])
            negative_nodes.extend(walk_list)

    sample_nodes = list(set(changable_nodes).union(negative_nodes))
    return sample_nodes


def k_hop_seq(g, i, sample_num, seq, search):
    if len(seq) == sample_num:
        return seq
    while len(seq) < sample_num:
        degrees = list()
        neighs = list()
        for nbr in g.neighbors(i):
            degrees.append(g.degree(nbr))
            neighs.append(nbr)
        if degrees == []:
            return
        d_sum = sum(degrees)
        degrees_prob = [x/d_sum for x in degrees]
        neighs_choice = random.choices(neighs, degrees_prob, k=1)
        if neighs_choice in seq:
            return
        seq.append(neighs_choice[0])
        search.append(neighs_choice[0])
        k_hop_seq(g, neighs_choice[0], sample_num, seq, search)
        search.pop()
    return seq

def compute_mad(model, data):
    final_feats = model.final_features(data.x, data.edge_index)
    final_feats = final_feats.detach().cpu().numpy()
    adj = to_dense_adj(data.edge_index).detach().cpu().numpy()
    m_trt = np.ones((data.num_nodes, data.num_nodes))
    target_idx = np.array([i for i in range(1, data.num_nodes+1)])
    mad = mad_value(in_arr=final_feats, mask_arr=m_trt, target_idx=target_idx)
    print('MAD: ', mad)
    return mad

def compute_dirichlet(model, data):
    final_feats = model.final_features(data.x, data.edge_index)
    final_feats = final_feats.detach().cpu().numpy()

    d_e = dirichlet_energy_regularizar(final_feats, data, 1)
    print('Dirichlet Energy: ', d_e)
    return d_e

def get_neighs_matrix(data, k):
    mtx = np.zeros((data.num_nodes, data.num_nodes))
    for i in range(data.num_nodes):
        _, subsets, _, _ = k_hop_subgraph(i, k, data.edge_index)
        mtx[subsets] = 1
    return mtx

def normalize(mx):
    # Row-normalize matrix
    # the output is the Tensor
    rowsum = mx.sum(1)
    r_inv = torch.pow(rowsum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0
    r_mat_inv = torch.diag(r_inv)
    mx = torch.matmul(r_mat_inv, mx)
    return mx

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger
