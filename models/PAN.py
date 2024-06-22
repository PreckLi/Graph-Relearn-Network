import torch
from torch_geometric.nn import PANConv
import torch.nn.functional as F
from torch_sparse import SparseTensor

class PAN(torch.nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self.hid_size = args.hid_size
        self.dropout = 0.5
        self.conv1 = PANConv(data.num_features, self.hid_size, 2)
        self.conv2 = PANConv(self.hid_size, data.num_classes, 2)


    def forward(self, x, edge_index):
        row, col = edge_index
        adj = SparseTensor(row=row, col=col, sparse_sizes=(x.shape[0], x.shape[0]))
        x, m1 = self.conv1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x, m2 = self.conv2(x, adj)
        result = F.log_softmax(x, -1)
        return result