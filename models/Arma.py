import torch
from torch_geometric.nn import ARMAConv
import torch.nn.functional as F


class ARMA(torch.nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self.hid_size = args.hid_size
        self.dropout = 0.5

        self.conv1 = ARMAConv(data.num_features, self.hid_size, num_stacks=2)
        self.conv2 = ARMAConv(self.hid_size, data.num_classes, num_stacks=2)


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