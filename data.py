import torch
from torch_geometric.data import Data
def get_random_graph(nodes, cutoff):

    positions = torch.randn(nodes, 3)

    distance_matrix = positions[:, None, :] - positions[None, :, :]
    distance_matrix = torch.linalg.norm(distance_matrix, dim=-1)
    assert distance_matrix.shape == (nodes, nodes)

    senders, receivers = torch.nonzero(distance_matrix < cutoff).T

    z = torch.zeros(len(positions), dtype=torch.int32)  # Create atomic_numbers tensor

    # Create edge index tensor by stacking senders and receivers
    edge_index = torch.stack([senders, receivers], dim=0)

    # Create a PyTorch Geometric Data object
    graph = Data(
        pos = positions,
        relative_vectors = positions[receivers] - positions[senders],
        numbers=z,
        edge_index=edge_index,
        num_nodes=len(positions)
    )

    return graph
