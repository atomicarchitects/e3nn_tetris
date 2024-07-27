import sys
sys.path.append("..")

from model import SimpleNetwork
from data import get_tetris, get_random_graph
import torch
from e3nn import o3

torch._dynamo.config.capture_scalar_outputs = True

graph = get_random_graph(nodes=5, cutoff=1.5)
model = SimpleNetwork(
    relative_vectors_irreps=o3.Irreps.spherical_harmonics(lmax=2),
    node_features_irreps=o3.Irreps("16x0e"),
)


args_in = (graph.y,
    graph.relative_vectors,
    graph.edge_index,
    graph.num_nodes)

es = torch.export.export(model, args_in)
print(es)


