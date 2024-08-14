import torch
torch.jit.script = lambda x: x
import torch.nn as nn
import e3nn
e3nn.set_optimization_defaults(jit_script_fx=False)

from e3nn import o3

from torch_scatter import scatter_mean, scatter_sum

class Layer(nn.Module):
    """A layer of a simple E(3)-equivariant message passing network."""
    
    denominator: int = 1.5
    
    def __init__(self,
                relative_positions_irreps: o3.Irreps,
                node_features_irreps: o3.Irreps,
                target_irreps: o3.Irreps):

      super().__init__()
      
      self.tp = o3.FullTensorProduct(relative_positions_irreps,
                                     node_features_irreps)
      
      tp_irreps = self.tp.irreps_out.regroup() + node_features_irreps

      self.linear = o3.Linear(irreps_in=tp_irreps,
                              irreps_out=target_irreps)
      
      self.shortcut = o3.Linear(irreps_in=node_features_irreps,
                                irreps_out=target_irreps)

    def forward(self,
                node_features,
                relative_positions_sh,
                senders, receivers) -> torch.Tensor:
    
        node_features_broadcasted = node_features[senders]
        
        # Resnet-style shortct
        shortcut_aggregated = scatter_mean(
            node_features_broadcasted,
            receivers.unsqueeze(1).expand(-1, node_features_broadcasted.size(dim=1)),
            dim=0,
            dim_size=node_features.shape[0]
        )
                
        shortcut = self.shortcut(shortcut_aggregated)
    
        # Tensor product of the relative vectors and the neighbouring node features.
        tp = self.tp(relative_positions_sh, node_features_broadcasted)
    
        # Construct message by appending to existing node_feature 
        messages = torch.cat([node_features_broadcasted, tp], dim=-1)
       
        # Aggregate the node features
        node_feats = scatter_mean(
            messages,
            receivers.unsqueeze(1).expand(-1, messages.size(dim=1)),
            dim=0,
            dim_size=node_features.shape[0]
        )
        
        
        # Normalize
        node_feats = node_feats / self.denominator
        
        # Apply linear
        node_feats = self.linear(node_feats)

        # Skipping scalar activation for now
        
        # Add shortcut to node_features
        node_features = node_feats + shortcut
        return node_features

class Model(torch.nn.Module):
  
  sh_lmax: int = 3

  def __init__(self):

      super().__init__()

      node_features_irreps = o3.Irreps("1x0e")
      relative_positions_irreps = o3.Irreps([(1,(l,(-1)**l)) for l in range(1,self.sh_lmax+1)])
      self.sph = o3.SphericalHarmonics(
        irreps_out=relative_positions_irreps,
        normalize=True, normalization="norm")
      
      
      output_irreps = [o3.Irreps("32x0e+8x1o+8x2e"), o3.Irreps("32x0e+8x1e+8x1o+8x2e+8x2o")] + [o3.Irreps("0o + 7x0e")]
      
      layers = []
      for target_irreps in output_irreps:
        layers.append(Layer(
                      relative_positions_irreps,
                      node_features_irreps,
                      target_irreps))
        node_features_irreps = target_irreps
      
      self.layers = torch.nn.ModuleList(layers)
          
  def forward(self, graphs):
    
    node_features, pos, edge_index, batch, num_nodes = graphs.numbers, graphs.pos, graphs.edge_index, graphs.batch, graphs.num_nodes
    senders, receivers = edge_index
    relative_positions= pos[receivers] - pos[senders]
    
    # Apply spherical harmonics
    relative_positions_sh = self.sph(relative_positions)
    
    for layer in self.layers:
      node_features = layer(
        node_features,
        relative_positions_sh,
        senders,
        receivers,
      )

    # Readout logits
    pred = scatter_sum(
        node_features,
        batch,
        dim=0,
        dim_size=8)  # [num_graphs, 1 + 7] = [8,8]
    odd, even1, even2 = pred[:, :1], pred[:, 1:2], pred[:, 2:]
    logits = torch.concatenate([odd * even1, -odd * even1, even2], dim=1)
    assert logits.shape == (8, 8)  # [num_graphs, num_classes]
    return logits