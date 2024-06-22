import torch
from torch_geometric.nn import GPSConv, GCNConv
from torch.nn import Linear
import torch.nn.functional as F

class GPS(torch.nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self.hid_size = args.hid_size
        self.dropout = args.dropout
        self.conv1 = GPSConv(data.num_features, GCNConv(data.num_features, data.num_features), heads=1, attn_dropout=args.dropout)
        self.conv2 = GPSConv(1*self.hid_size, GCNConv(self.hid_size, self.hid_size), heads=1, attn_dropout=args.dropout)
        self.l1 = Linear(data.num_features, self.hid_size)
        self.l2 = Linear(self.hid_size, data.num_classes)


    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.l1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.l2(x)
        result = F.log_softmax(x, -1)
        return result

    def final_features(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.l1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv2(x, edge_index)
        return x