import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from rough.code.gcn_gru_model import RecurrentGCNGRU
from rough.code.utils import normalize_adjacency


class CustomGCNGRUFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, max_num_nodes, num_k_paths, num_preprocess_out=512, batch_size=32, features_dim=32):
        super(CustomGCNGRUFeatureExtractor, self).__init__(observation_space, features_dim)

        gru_hidden_size = 1024
        num_gru_layers = 1
        self.batch_size = batch_size
        self.max_num_nodes = max_num_nodes
        self.num_k_paths = num_k_paths

        # Check if GPU is available and select the device accordingly
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initialize preprocessing_model and linear layers, and move them to the selected device
        self.preprocessing_model = RecurrentGCNGRU(num_preprocess_out, gru_hidden_size, num_gru_layers).to(self.device)
        self.linear = nn.Linear(gru_hidden_size, num_preprocess_out).to(self.device)

    def forward(self, observations):
        # Move observations to the same device as the model

        observations = observations.to(self.device)

        adjacency_matrix = observations[:, :1000000].view(-1, self.max_num_nodes, self.max_num_nodes)
        node_features = observations[:, 1000000:1005000].view(-1, self.max_num_nodes, self.num_k_paths)
        remaining_observations = observations[:, 1005000:]

        normalized_adjacency_matrix = normalize_adjacency(adjacency_matrix, self.max_num_nodes, self.batch_size)

        gcn_gru_output = self.preprocessing_model(node_features, normalized_adjacency_matrix)

        final_preprocessed_output = torch.relu(self.linear(gcn_gru_output))

        flattened_obs = torch.cat(
            (
                final_preprocessed_output,
                remaining_observations
            ),
            dim=1)

        return flattened_obs
