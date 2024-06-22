import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch_geometric.nn import GCNConv


class Predlayer(nn.Module):
    def __init__(self, input_size, out_size):
        super(Predlayer, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        self.weight = nn.Parameter(torch.FloatTensor(out_size, input_size))
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, aggregate_feats):
        combined = aggregate_feats
        combined = F.relu(self.weight.mm(combined.t())).t()
        return combined


class Supporter(nn.Module):
    def __init__(self, input_size, hid_size, out_size, agg_func='MEAN'):
        super(Supporter, self).__init__()
        self.predlayer1 = Predlayer(input_size, hid_size)
        self.predlayer2 = Predlayer(hid_size, out_size)
        self.dropout = 0.5
        self.agg_func = agg_func

    def forward(self, x):
        x = self.predlayer1(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.predlayer2(x)
        result = F.log_softmax(x, -1)
        return result


def aggregate(x, neighs_list, agg_func='MEAN'):
    nodes = [i for i in range(x.shape[0])]
    embed_matrix = x
    mask = torch.zeros(len(neighs_list), len(nodes))
    column_indices = [nodes[n] for samp_neigh in neighs_list for n in samp_neigh]
    row_indices = [i for i in range(len(neighs_list)) for j in range(len(neighs_list[i]))]
    mask[row_indices, column_indices] = 1
    # mask = mask - torch.eye(row_indices, column_indices)
    if agg_func == 'MEAN':
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh).to(embed_matrix.device)
        aggregate_feats = mask.mm(embed_matrix)
    elif agg_func == 'MAX':
        indexs = [x.nonzero() for x in mask == 1]
        aggregate_feats = []
        for feat in [embed_matrix[x.squeeze()] for x in indexs]:
            if len(feat.size()) == 1:
                aggregate_feats.append(feat.view(1, -1))
            else:
                aggregate_feats.append(torch.max(feat, 0)[0].view(1, -1))
        aggregate_feats = torch.cat(aggregate_feats, 0)
    return aggregate_feats

class Supporter2(nn.Module):
    def __init__(self, input_size, hid_size, out_size, dropout=0.2):
        super(Supporter2, self).__init__()
        self.predlayer1 = GCNConv(input_size, hid_size)
        self.predlayer2 = GCNConv(hid_size, out_size)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.predlayer1(x, edge_index)
        x = F.leaky_relu(x, 0.2)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.predlayer2(x, edge_index)
        result = F.log_softmax(x, -1)
        return result