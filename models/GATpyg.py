import numpy as np
import torch
from torch_geometric.nn import GATConv
import torch.nn as nn
import torch.nn.functional as F


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.alpha = alpha
        self.dropout = dropout
        self.conv1 = GATConv(nfeat, nhid, nheads, dropout=dropout)
        self.conv2 = GATConv(nheads * nhid, nclass, 1, dropout=dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(x, alpha=self.alpha)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def final_features(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv2(x, edge_index)
        return x
