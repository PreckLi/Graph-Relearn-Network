from torch.nn import Linear
from torch.nn import functional as F

import torch
from torch import Tensor
from torch.nn import Parameter, LeakyReLU
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.conv import MessagePassing, GCNConv, GATConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.inits import zeros, glorot
from torch_geometric.typing import OptTensor


class GENLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, num_stacks=1, num_layers=1, dropout=0.0, share_weights=False, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_stacks = num_stacks
        self.num_layers = num_layers
        self.share_weights = share_weights
        self.dropout = dropout
        self.act = LeakyReLU(negative_slope=0.2)

        K, T, F_in, F_out = num_stacks, num_layers, in_channels, out_channels
        T = 1 if self.share_weights else T
        self.weight = Parameter(torch.Tensor(max(1, T - 1), K, F_out, F_out))
        self.init_weight = Parameter(torch.Tensor(K, F_in, F_out))
        self.root_weight = Parameter(torch.Tensor(T, K, F_in, F_out))
        self.dense_weight = Parameter(torch.Tensor(T, K, F_out, F_out))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        if not isinstance(self.init_weight, torch.nn.UninitializedParameter):
            glorot(self.init_weight)
            glorot(self.root_weight)
            glorot(self.dense_weight)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        """"""

        if isinstance(edge_index, Tensor):
            edge_index, edge_weight = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim),
                add_self_loops=False, flow=self.flow, dtype=x.dtype)

        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim),
                add_self_loops=False, flow=self.flow, dtype=x.dtype)

        x = x.unsqueeze(-3)

        out = x
        for t in range(self.num_layers):
            if t == 0:
                out = out @ self.init_weight
                x_outsize_list = [out]
            else:
                out = out @ self.weight[0 if self.share_weights else t - 1]

            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            out = self.propagate(edge_index, x=out, edge_weight=edge_weight,
                                 size=None)
            if t != 0:
                dense_root = F.dropout(x_outsize_list[-1], p=self.dropout, training=self.training)
                dense_root = dense_root.mean(dim=-3).unsqueeze(-3)
                dense_root = dense_root @ self.dense_weight[0 if self.share_weights else t]
                x_outsize_list.append(dense_root)

            root = F.dropout(x, p=self.dropout, training=self.training)
            root = root @ self.root_weight[0 if self.share_weights else t]
            if t != 0:
                out = out + root + 0.6 * dense_root
            else:
                out = out + root

            if self.bias is not None:
                out = out + self.bias[0 if self.share_weights else t]

            if self.act is not None:
                out = self.act(out)

        return out.sum(dim=-3)

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)


class GRN(torch.nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self.hid_size = args.hid_size
        self.dropout = args.dropout
        if args.gnn_style == 'grn_gcn':
            self.conv1 = GCNConv(data.num_features, self.hid_size)
            self.conv2 = GCNConv(self.hid_size, self.hid_size)
            self.conv3 = GCNConv(self.hid_size, data.num_classes)
        if args.gnn_style == 'grn_gat':
            self.conv1 = GATConv(data.num_features, self.hid_size, heads=args.gat_heads, dropout=self.dropout)
            self.conv2 = GATConv(args.gat_heads * self.hid_size, self.hid_size, heads=1, dropout=self.dropout)
            self.conv3 = GATConv(self.hid_size, data.num_classes, heads=1, dropout=self.dropout)
        else:
            self.conv1 = GENLayer(data.num_features, self.hid_size, 2, 2)
            self.conv2 = GENLayer(self.hid_size, self.hid_size, 2, 2)
            self.conv3 = GENLayer(self.hid_size, self.hid_size, 2, 2)
            self.conv4 = GENLayer(self.hid_size, self.hid_size, 2, 2)
            self.conv5 = GENLayer(self.hid_size, self.hid_size, 2, 2)
            self.conv6 = GENLayer(self.hid_size, data.num_classes, 1, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv6(x, edge_index)
        result = F.log_softmax(x, -1)
        return result

    def final_features(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv6(x, edge_index)
        return x
