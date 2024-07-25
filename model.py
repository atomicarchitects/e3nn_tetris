import torch
import torch.nn as nn
import e3nn
e3nn.set_optimization_defaults(jit_script_fx=False)

from e3nn import o3
# from torch_runstats.scatter import scatter_mean
from utils import scatter_mean


class AtomEmbedding(nn.Module):
    """Embeds atomic atomic_numbers into a learnable vector space."""

    def __init__(self, embed_dims: int, max_atomic_number: int):
        super().__init__()
        self.embed_dims = embed_dims
        self.max_atomic_number = max_atomic_number
        self.embedding = nn.Embedding(num_embeddings=max_atomic_number, embedding_dim=embed_dims)

    def forward(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        atom_embeddings = self.embedding(atomic_numbers)
        return atom_embeddings
    
class MLP(nn.Module):

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        hidden_dims: int = 32,
        num_layers: int = 2):

        super(MLP, self).__init__()
        layers = []
        for i in range(num_layers - 1):
            layers.append(nn.Linear(input_dims if i == 0 else hidden_dims, hidden_dims))
            layers.append(nn.LayerNorm(hidden_dims))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dims, output_dims))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
      return self.model(x)

class SimpleNetwork(nn.Module):
    """A layer of a simple E(3)-equivariant message passing network."""

    sh_lmax: int = 2
    lmax: int = 2
    init_node_features: int = 16
    max_atomic_number: int = 12
    num_hops: int = 2
    output_dims: int = 1

    def __init__(self,
                relative_vectors_irreps: o3.Irreps,
                node_features_irreps: o3.Irreps):

      super().__init__()

      self.embed = AtomEmbedding(self.init_node_features, self.max_atomic_number)
      self.sph = o3.SphericalHarmonics(irreps_out=o3.Irreps.spherical_harmonics(self.sh_lmax), normalize=True, normalization="norm")
      
      
      # Currently hardcoding 1 layer
      
      print("node_features_irreps", node_features_irreps)
      
      self.tp = o3.FullTensorProduct(relative_vectors_irreps.regroup(),
                                     node_features_irreps.regroup(),
                                    filter_ir_out=[o3.Irrep(f"{l}e") for l in range(self.lmax+1)] + [o3.Irrep(f"{l}o") for l in range(self.lmax+1)])
      self.linear = o3.Linear(irreps_in=self.tp.irreps_out.regroup(), irreps_out=self.tp.irreps_out.regroup())
      print("TP+Linear", self.linear.irreps_out)
      self.mlp = MLP(input_dims =  1, # Since we are inputing the norms will always be (..., 1)
                     output_dims = self.tp.irreps_out.num_irreps)


      self.elementwise_tp = o3.ElementwiseTensorProduct(o3.Irreps(f"{self.tp.irreps_out.num_irreps}x0e"), self.linear.irreps_out.regroup())
      print("node feature broadcasted", self.elementwise_tp.irreps_out)

      # Poor mans filter function (Can already feel the judgement). Replicating irreps_array.filter("0e")
      self.filter_tp = o3.FullTensorProduct(self.tp.irreps_out.regroup(), o3.Irreps("0e"), filter_ir_out=[o3.Irrep("0e")])
      self.register_buffer("dummy_input", torch.ones(1))

      print("aggregated node features", self.filter_tp.irreps_out)

      self.readout_mlp = MLP(input_dims = self.filter_tp.irreps_out.num_irreps,
                             output_dims = self.output_dims)

    def forward(self,
                numbers: torch.Tensor,
                relative_vectors: torch.Tensor,
                edge_index: torch.Tensor,
                num_nodes: int) -> torch.Tensor:

        node_features = self.embed(numbers)
        relative_vectors = relative_vectors
        senders, receivers = edge_index

        relative_vectors_sh = self.sph(relative_vectors)
        relative_vectors_norm = torch.linalg.norm(relative_vectors, axis=-1, keepdims=True)


        # Currently harcoding 1 hop


        # Layer 0
    
        # Tensor product of the relative vectors and the neighbouring node features.
        node_features_broadcasted = node_features[senders]

        tp = self.tp(relative_vectors_sh, node_features_broadcasted)


        # Apply linear
        tp = self.linear(tp)


        # Simply multiply each irrep by a learned scalar, based on the norm of the relative vector.
        scalars = self.mlp(relative_vectors_norm)
        node_features_broadcasted = self.elementwise_tp(scalars, tp)


        # Aggregate the node features back.
        node_features = scatter_mean(
            node_features_broadcasted,
            receivers,
            dim = node_features.shape[0]
        )

        # # Global readout.

        # Filter out 0e
        node_features = self.filter_tp(node_features, self.dummy_input)

        graph_globals = scatter_mean(node_features, output_dim=[num_nodes])
        return self.readout_mlp(graph_globals)