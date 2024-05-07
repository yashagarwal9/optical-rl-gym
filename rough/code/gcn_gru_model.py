import torch
import torch.nn as nn
from torch.nn.modules.module import Module

from rough.code.gcn_layer import GCNLayer


class RecurrentGCNGRU(Module):
    def __init__(self, num_output_features, gru_hidden_size, num_gru_layers):
        super(RecurrentGCNGRU, self).__init__()
        self.num_gcn_layers = 5

        # Check if GPU is available and select the device accordingly
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initialize GCN layers and move them to the selected device
        self.gcn_layers = nn.ModuleList(
            [GCNLayer(1, num_output_features).to(self.device) for _ in range(self.num_gcn_layers)]
        )

        # Initialize GRU layer and move it to the selected device
        self.gru = nn.GRU(num_output_features, gru_hidden_size, num_gru_layers, batch_first=True).to(self.device)

    def forward(self, node_feature_matrices, adj_matrix):
        # Move inputs to the same device as the model
        node_feature_matrices = node_feature_matrices.to(self.device)
        adj_matrix = adj_matrix.to(self.device)

        gcn_outputs = []
        for i, gcn_layer in enumerate(self.gcn_layers):
            # Apply each GCN layer to its corresponding node feature matrix
            node_features = node_feature_matrices[:, :, i].unsqueeze(-1)
            gcn_output = gcn_layer(node_features, adj_matrix)
            gcn_outputs.append(gcn_output)

        # Concatenate the outputs along the sequence dimension
        gcn_sequence = torch.cat(gcn_outputs, dim=1)

        # Pass the sequence of GCN outputs to the GRU
        gru_output, _ = self.gru(gcn_sequence)

        gru_output = gru_output[:, -1, :]  # Take the last time step's output

        return gru_output
