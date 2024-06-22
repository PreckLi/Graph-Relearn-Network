import torch
from torch_geometric.nn import ChebConv
import torch.nn.functional as F


class Chebnet(torch.nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self.hid_size = args.hid_size
        self.dropout = 0.5
        self.conv1 = ChebConv(data.num_features, self.hid_size, 2)
        self.conv2 = ChebConv(self.hid_size, data.num_classes, 2)


    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv2(x, edge_index)
        result = F.log_softmax(x, -1)
        return result

    def final_features(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv2(x, edge_index)
        return x