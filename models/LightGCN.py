import torch
from torch_geometric.nn import LGConv, Linear
import torch.nn.functional as F


class LightGCN(torch.nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self.hid_size = args.hid_size
        self.dropout = 0.5

        self.conv1 = LGConv()
        self.conv2 = LGConv()
        self.lin1 = self.lin1 = Linear(data.num_node_features, self.hid_size)
        self.lin2 = Linear(self.hid_size, data.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.lin2(x)
        result = F.log_softmax(x, -1)
        return result

    def final_features(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv2(x, edge_index)
        return x