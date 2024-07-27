import sys
sys.path.append("..")

from model import SimpleNetwork
from data import get_tetris, get_random_graph
import torch
from e3nn import o3

torch._dynamo.config.capture_scalar_outputs = True

graphs = get_tetris()

model = SimpleNetwork(
    relative_vectors_irreps=o3.Irreps.spherical_harmonics(lmax=2),
    node_features_irreps=o3.Irreps("16x0e"),
)

optimizer = torch.optim.Adam(model.parameters())

def loss_fn(graphs):
    logits = model(graphs.numbers,
                   graphs.pos,
                   graphs.edge_index,
                   graphs.num_nodes)
    labels = graphs.y.unsqueeze(-1).float()  # [num_graphs]
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss, logits


def apply_random_rotation(graphs):
    """Apply a random rotation to the nodes of the graph."""
    alpha, beta, gamma = torch.rand(3) * 2 * torch.pi - torch.pi

    rotated_pos = o3.angles_to_matrix(alpha, beta, gamma) @ graphs.pos.T
    rotated_pos = rotated_pos.T

    rotated_graphs = graphs.clone()
    rotated_graphs.pos = rotated_pos
    return rotated_graphs

model.train()
for _ in range(10):
    
    graphs = apply_random_rotation(graphs)
    optimizer.zero_grad()
    loss, logits = loss_fn(graphs)
    loss.backward()
    optimizer.step()

    preds = torch.argmax(logits, dim=1)
    accuracy = (preds == graphs.y.squeeze()).float().mean()

# es = torch.export.export(model,
#                         (graphs.numbers,
#                         graphs.pos,
#                         graphs.edge_index,
#                         graphs.num_nodes))
# print(es)


