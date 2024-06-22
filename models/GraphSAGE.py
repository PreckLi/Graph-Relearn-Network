import torch
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F

class SAGEConv2(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=1):
        super(SAGEConv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.weight = torch.nn.Parameter(torch.Tensor(in_channels, out_channels * heads))
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        self.attention = torch.nn.Parameter(torch.Tensor(2 * out_channels, 1 * heads))
        self.reset_parameters()
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.zeros_(self.bias)
        torch.nn.init.xavier_uniform_(self.attention)

    def forward(self, x, edge_index):
        device = torch.device("cuda:0")
        adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]).to(device), (x.shape[0], x.shape[0]))
        deg = torch.sparse.sum(adj, dim=1).to_dense()
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt * torch.sparse.mm(adj, deg_inv_sqrt.unsqueeze(1))
        x_j = torch.sparse.mm(norm, x @ self.weight).view(-1, self.heads, self.out_channels)
        x_i = self.lin(x).view(-1, self.heads, self.out_channels)
        alpha = F.leaky_relu(torch.cat([x_i.repeat(1, 1, x_j.shape[1]), x_j.repeat(1, x_i.shape[1], 1)], dim=-1) @ self.attention, negative_slope=0.2)
        alpha = alpha.squeeze().squeeze()
        alpha = F.dropout(alpha, p=0.6, training=self.training)
        x = x_j.view(-1, self.out_channels * self.heads) @ alpha
        x = F.relu(x + self.bias)
        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self.hid_size = args.hid_size
        self.dropout = 0.5
        self.conv1 = SAGEConv(data.num_features, args.hid_size)
        self.conv2 = SAGEConv(args.hid_size, data.num_classes)

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