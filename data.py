import torch
from torch_geometric.data import Data, Batch
from torch_geometric.nn import radius_graph

def get_random_graph(nodes, cutoff) -> Data:

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
        y=z,
        edge_index=edge_index,
        num_nodes=len(positions)
    )

    return graph

def get_tetris() -> Batch:
    pos = [
        [[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0]],  # chiral_shape_1
        [[1, 1, 1], [1, 1, 2], [2, 1, 1], [2, 0, 1]],  # chiral_shape_2
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],  # square
        [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]],  # line
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]],  # corner
        [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0]],  # L
        [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 1]],  # T
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0]],  # zigzag
    ]
    pos = torch.tensor(pos, dtype=torch.float32)
    labels = torch.arange(8)

    graphs = []

    for p, l in zip(pos, labels):
        edge_index = radius_graph(p, r=1.1)
        data = Data(pos=pos,
                    y=l,
                    edge_index = edge_index,
                    relative_vectors = p[edge_index[0]] - p[edge_index[1]],
                    num_nodes = 4)

        graphs.append(data)

    batch = Batch.from_data_list(graphs)
    batch.pos = batch.pos.view(-1, 3)
    batch.relative_vectors = batch.relative_vectors.view(-1, 3)
    
    return batch 