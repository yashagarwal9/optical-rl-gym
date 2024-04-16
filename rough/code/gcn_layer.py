import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
import math


class GCNLayer(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, node_features, adjacency_matrix):
        # node_features shape: (batch_size, num_nodes, in_features)
        # adjacency_matrix shape: (batch_size, num_nodes, num_nodes)

        # support = torch.mm(input_tensor, self.weight)
        # output_tensor = torch.spmm(adj, support)

        # Apply the linear transformation to node features
        transformed_features = torch.matmul(node_features, self.weight)

        # Multiply transformed features by the normalized adjacency matrix
        output_features = torch.bmm(adjacency_matrix, transformed_features)

        if self.bias is not None:
            return output_features + self.bias
        else:
            return output_features

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'
