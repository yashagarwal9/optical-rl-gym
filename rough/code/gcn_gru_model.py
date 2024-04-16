import torch
import torch.nn as nn
from torch.nn.modules.module import Module

from rough.code.gcn_layer import GCNLayer


class RecurrentGCNGRU(Module):
    def __init__(self, num_output_features, gru_hidden_size, num_gru_layers):
        super(RecurrentGCNGRU, self).__init__()
        self.num_gcn_layers = 5
        self.gcn_layers = nn.ModuleList(
            [GCNLayer(1, num_output_features) for _ in range(self.num_gcn_layers)]
        )
        self.gru = nn.GRU(num_output_features, gru_hidden_size, num_gru_layers, batch_first=True)

    def forward(self, node_feature_matrices, adj_matrix):
        # node_feature_matrices is expected to be a list of feature matrices,
        # one for each GCN layer.
        gcn_outputs = []
        for i, gcn_layer in enumerate(self.gcn_layers):
            # Apply each GCN layer to its corresponding node feature matrix
            node_features = node_feature_matrices[:, :, i].unsqueeze(-1)
            gcn_output = gcn_layer(node_features, adj_matrix)
            gcn_outputs.append(gcn_output)
            # gcn_outputs.append(gcn_output.unsqueeze(1))  # Add sequence dimension

        # Concatenate the outputs along the sequence dimension
        gcn_sequence = torch.cat(gcn_outputs, dim=1)

        # Pass the sequence of GCN outputs to the GRU
        gru_output, _ = self.gru(gcn_sequence)

        gru_output = gru_output[:, -1, :]  # Take the last time step's output

        return gru_output
