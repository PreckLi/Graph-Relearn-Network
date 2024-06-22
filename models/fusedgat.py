import torch
from torch_geometric.nn import FusedGATConv
import torch.nn.functional as F

class FusedGAT(torch.nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self.hid_size = args.hid_size
        self.dropout = 0.5
        self.conv1 = FusedGATConv(data.num_features, self.hid_size, heads=args.gat_heads, add_self_loops=False)
        self.conv2 = FusedGATConv(args.gat_heads * self.hid_size, data.num_features, heads=1, add_self_loops=False)


    def forward(self, x, edge_index):
        csr, csc, perm = FusedGATConv.to_graph_format(edge_index, size=(x.shape[0], x.shape[0]))
        x = self.conv1(x, csr, csc, perm)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv2(x, csr, csc, perm)
        result = F.log_softmax(x, -1)
        return result

    def final_features(self, x, edge_index):
        csr, csc, perm = FusedGATConv.to_graph_format(edge_index, size=(x.shape[0], x.shape[0]))
        x = self.conv1(x, csr, csc, perm)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv2(x, csr, csc, perm)
        return x