import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math


class GCNLayer(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Check if GPU is available and select the device accordingly
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initialize parameters and move them to the selected device
        self.weight = Parameter(torch.FloatTensor(in_features, out_features)).to(self.device)

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features)).to(self.device)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, node_features, adjacency_matrix):
        # Move inputs to the same device as parameters
        node_features = node_features.to(self.device)
        adjacency_matrix = adjacency_matrix.to(self.device)

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
