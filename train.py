import sys
sys.path.append("..")

from model import SimpleNetwork
from data import get_random_graph
import torch
from e3nn import o3

graph = get_random_graph(5, 2.5)
model = SimpleNetwork(
    relative_vectors_irreps=o3.Irreps.spherical_harmonics(lmax=2),
    node_features_irreps=o3.Irreps("16x0e"),
)

# Currently turning off since Linear still needs weights
# Also need confirm that the model is working
model = torch.compile(model, fullgraph=True, disable=True)


model(graph.numbers,
      graph.relative_vectors,
      graph.edge_index,
      graph.num_nodes)