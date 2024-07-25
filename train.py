import sys
sys.path.append("..")

from model import SimpleNetwork
from data import get_random_graph
import torch
from e3nn import o3

torch._dynamo.config.capture_scalar_outputs = True

graph = get_random_graph(5, 2.5)
model = SimpleNetwork(
    relative_vectors_irreps=o3.Irreps.spherical_harmonics(lmax=2),
    node_features_irreps=o3.Irreps("16x0e"),
)


model = torch.compile(model, fullgraph=True)

# model(graph.numbers,
#     graph.relative_vectors,
#     graph.edge_index,
#     graph.num_nodes)

# Currently turning off since Linear still needs weights
# Also need confirm that the model is working
model = torch.export.export(model,
                            (graph.numbers,
                            graph.relative_vectors,
                            graph.edge_index,
                            graph.num_nodes))


