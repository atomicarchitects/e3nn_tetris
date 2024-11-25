import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import radius_graph

def get_random_graph(n_nodes, cutoff) -> Data:

    positions = torch.randn(n_nodes, 3)

    distance_matrix = positions[:, None, :] - positions[None, :, :]
    distance_matrix = torch.linalg.norm(distance_matrix, dim=-1)
    assert distance_matrix.shape == (n_nodes, n_nodes)

    senders, receivers = torch.nonzero(distance_matrix < cutoff).T

    z = torch.zeros(len(positions), dtype=torch.int32)  # Create atomic_numbers tensor

    # Create edge index tensor by stacking senders and receivers
    edge_index = torch.stack([senders, receivers], dim=0)

    # Create a PyTorch Geometric Data object
    graph = Data(
        pos = positions, # node positions
        relative_vectors = positions[receivers] - positions[senders], # node relative positions
        y=z, # graph label
        edge_index=edge_index, # edge indices
        numbers=torch.ones((len(positions),1)),  # node features
    )

    return graph

def get_tetris():
    """Get the Tetris dataset."""
    
    all_positions = [
        [[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0]],  # chiral_shape_1
        [[1, 1, 1], [1, 1, 2], [2, 1, 1], [2, 0, 1]],  # chiral_shape_2
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],  # square
        [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]],  # line
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]],  # corner
        [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0]],  # L
        [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 1]],  # T
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0]],  # zigzag
    ]
    all_positions = torch.tensor(all_positions, dtype=torch.float32)
    all_labels = torch.arange(8)
    
    graphs = []
    for positions, label in zip(all_positions, all_labels):
        edge_index = radius_graph(positions, r=1.1)
        senders, receivers = edge_index
        
        data = Data(
            numbers=torch.ones((len(positions),1)),  # node features
            pos=positions,  # node positions
            edge_index=edge_index,  # edge indices
            y=label  # graph label
        )
        graphs.append(data)
    
    return next(iter(DataLoader(graphs, batch_size=len(graphs))))
