import torch
from scipy.sparse import coo_matrix
import numpy as np


def normalize_adjacency(adj, num_nodes, batch_size):
    """Normalize the adjacency matrix."""

    # Add self loops
    # identity_batch = torch.eye(num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)
    # adj = torch.where(
    #     adj == 0,
    #     adj + identity_batch,
    #     adj
    # )

    identity = torch.eye(num_nodes).repeat(batch_size, 1, 1)
    adj = adj + identity

    degree_matrix = adj.sum(dim=-1)
    # Compute the D^{-1/2} component
    degree_matrix_inv_sqrt = degree_matrix.pow(-0.5)
    degree_matrix_inv_sqrt[degree_matrix_inv_sqrt == float('inf')] = 0
    degree_matrix_inv_sqrt = degree_matrix_inv_sqrt.unsqueeze(-1) * torch.eye(num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)

    # Perform the symmetric normalization
    normalized_adjacency_matrix = torch.matmul(torch.matmul(degree_matrix_inv_sqrt, adj), degree_matrix_inv_sqrt)

    # Make sure to handle NaNs that might occur during division by zero
    normalized_adjacency_matrix[torch.isnan(normalized_adjacency_matrix)] = 0

    return normalized_adjacency_matrix.to(dtype=torch.float32)

    # # Normalize
    # adj = coo_matrix(adj)
    # row_sum = np.array(adj.sum(1))
    # d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # d_mat_inv_sqrt = coo_matrix((d_inv_sqrt, (np.arange(len(d_inv_sqrt)), np.arange(len(d_inv_sqrt)))), shape=adj.shape)
    #
    # normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    #
    # # Convert to torch sparse tensor
    # indices = torch.from_numpy(np.vstack((normalized_adj.row, normalized_adj.col)).astype(np.int64))
    # values = torch.from_numpy(normalized_adj.data.astype(np.float32))
    # shape = torch.Size(normalized_adj.shape)
    # return torch.sparse.FloatTensor(indices, values, shape)


# def adjust_tensor_size(input_tensor, row_limit, col_limit):
#     # Get the current shape of the input tensor
#     current_shape = input_tensor.shape
#
#     # If the tensor has more rows than row_limit, trim it
#     if current_shape[1] > row_limit:
#         input_tensor = input_tensor[:, row_limit, :]
#
#     # If the tensor has fewer rows than row_limit, pad it with zeros
#     elif current_shape[1] < row_limit:
#         padding_rows = row_limit - current_shape[1]
#         padding = torch.zeros(padding_rows, current_shape[2])
#         input_tensor = torch.cat((input_tensor, padding), dim=1)
#
#     # If the tensor has more columns than col_limit, trim it
#     if current_shape[2] > col_limit:
#         input_tensor = input_tensor[:, :, :col_limit]
#
#     # If the tensor has fewer columns than col_limit, pad it with zeros
#     elif current_shape[2] < col_limit:
#         padding_columns = col_limit - current_shape[2]
#         padding = torch.zeros(row_limit, padding_columns)
#         input_tensor = torch.cat((input_tensor, padding), dim=2)
#
#     return input_tensor


def adjust_tensor_size(input_tensor, row_limit, col_limit):
    batch_size, current_rows, current_cols = input_tensor.shape

    # Initialize a tensor filled with zeros with the desired shape
    adjusted_tensor = np.zeros((batch_size, row_limit, col_limit), dtype=np.float32)

    # Determine the number of rows and columns to copy from the original tensor
    rows_to_copy = min(current_rows, row_limit)
    cols_to_copy = min(current_cols, col_limit)

    # Copy the data from the input tensor to the adjusted tensor
    adjusted_tensor[:, :rows_to_copy, :cols_to_copy] = input_tensor[:, :rows_to_copy, :cols_to_copy]

    return torch.from_numpy(adjusted_tensor)
