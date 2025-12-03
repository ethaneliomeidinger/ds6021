import numpy as np

def get_adjacency(matrix, mode='topk', k=30, threshold=0.0, symmetrize=True):
    """
    Convert a connectivity matrix into an adjacency matrix.

    Parameters:
        matrix (np.ndarray): Input connectivity matrix (N x N).
        mode (str): 'topk' or 'threshold'
        k (int): Number of top connections to retain (if mode='topk')
        threshold (float): Value threshold (if mode='threshold')
        symmetrize (bool): Make the adjacency matrix symmetric

    Returns:
        adj (np.ndarray): Processed adjacency matrix (N x N)
    """
    N = matrix.shape[0]
    adj = np.zeros_like(matrix)

    if mode == 'topk':
        for i in range(N):
            row = matrix[i]
            topk_idx = np.argsort(row)[-k:]  # get top-k indices
            adj[i, topk_idx] = matrix[i, topk_idx]
    elif mode == 'threshold':
        adj = np.where(matrix > threshold, matrix, 0)
    else:
        raise ValueError("Unsupported mode. Choose 'topk' or 'threshold'.")

    if symmetrize:
        adj = np.maximum(adj, adj.T)

    return adj