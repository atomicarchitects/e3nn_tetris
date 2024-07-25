import torch
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import RadiusGraph

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
    radius_graph = RadiusGraph(r=1.1)

    for p, l in zip(pos, labels):
        data = Data(pos=p, y=l)
        data = radius_graph(data)

        graphs.append(data)

    return Batch.from_data_list(graphs)