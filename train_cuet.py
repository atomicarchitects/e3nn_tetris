import time
import os
import nvtx

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.fx.experimental.proxy_tensor import make_fx
from tqdm.auto import tqdm
from e3nn import o3
from e3nn.util.jit import prepare

from torch_geometric.data import Data
from torch_scatter import scatter_mean, scatter_sum
import cuequivariance as cue
import cuequivariance_torch as cuet
import cuequivariance.segmented_tensor_product as stp
from cuequivariance.irreps_array.irrep_utils import into_list_of_irrep


from typing import Sequence, Optional
import itertools

torch.manual_seed(0)

# Borrowed from https://github.com/pytorch-labs/gpt-fast/blob/db7b273ab86b75358bd3b014f1f022a19aba4797/generate.py#L16-L18
torch.set_float32_matmul_precision("high")
import torch._dynamo.config
import torch._inductor.config

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
    
from data import get_tetris, get_random_graph

device = "cuda" if torch.cuda.is_available() else "cpu"


def full_tensor_product(
    irreps1: cue.Irreps,
    irreps2: cue.Irreps,
    irreps3_filter: Optional[Sequence[cue.Irrep]] = None,
) -> cue.EquivariantTensorProduct:
    """
    subscripts: ``lhs[iu],rhs[jv],output[kuv]``

    Construct a weightless channelwise tensor product descriptor.

    .. currentmodule:: cuequivariance

    Args:
        irreps1 (Irreps): Irreps of the first operand.
        irreps2 (Irreps): Irreps of the second operand.
        irreps3_filter (sequence of Irrep, optional): Irreps of the output to consider.

    Returns:
        EquivariantTensorProduct: Descriptor of the full tensor product.
    """
    G = irreps1.irrep_class

    if irreps3_filter is not None:
        irreps3_filter = into_list_of_irrep(G, irreps3_filter)

    d = stp.SegmentedTensorProduct.from_subscripts("iu,jv,kuv+ijk")

    for mul, ir in irreps1:
        d.add_segment(0, (ir.dim, mul))
    for mul, ir in irreps2:
        d.add_segment(1, (ir.dim, mul))

    irreps3 = []

    for (i1, (mul1, ir1)), (i2, (mul2, ir2)) in itertools.product(
        enumerate(irreps1), enumerate(irreps2)
    ):
        for ir3 in ir1 * ir2:
            # for loop over the different solutions of the Clebsch-Gordan decomposition
            for cg in cue.clebsch_gordan(ir1, ir2, ir3):
                d.add_path(i1, i2, None, c=cg)

                irreps3.append((mul1 * mul2, ir3))

    irreps3 = cue.Irreps(G, irreps3)
    irreps3, perm, inv = irreps3.sort()
    d = d.permute_segments(2, inv)

    d = d.normalize_paths_for_operand(-1)
    return cue.EquivariantTensorProduct(
        d,
        [irreps1, irreps2, irreps3],
        layout=cue.ir_mul,
    )


class Layer(nn.Module):
    """A layer of a simple E(3)-equivariant message passing network."""
    
    denominator: int = 1.5
    
    def __init__(self,
                relative_positions_irreps: cue.Irreps,
                node_features_irreps: cue.Irreps,
                target_irreps: cue.Irreps):

      super().__init__()
      
      e_tp = full_tensor_product(
          relative_positions_irreps,
          node_features_irreps,
      )
      self.tp = cuet.EquivariantTensorProduct(
          e_tp,
          layout=cue.ir_mul,
          optimize_fallback=True
      )
      
      tp_irreps = e_tp.output.irreps + node_features_irreps

      self.linear = cuet.Linear(irreps_in=tp_irreps,
                              irreps_out=target_irreps,
                                layout=cue.ir_mul,
                                optimize_fallback=True)
      
      self.shortcut = cuet.Linear(irreps_in=node_features_irreps,
                                irreps_out=target_irreps,
                                layout=cue.ir_mul,
                                optimize_fallback=True)

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

      node_features_irreps = cue.Irreps("O3", "1x0e")
      relative_positions_irreps = cue.Irreps("O3", o3.Irreps([(1,(l,(-1)**l)) for l in range(1,self.sh_lmax+1)]).__str__())
      
      output_irreps = [cue.Irreps("O3", "32x0e+8x1o+8x2e"), cue.Irreps("O3", "32x0e+8x1e+8x1o+8x2e+8x2o")] + [cue.Irreps("O3", "0o + 7x0e")]
      
      self.sph = cuet.EquivariantTensorProduct(
                    cue.descriptors.spherical_harmonics(cue.SO3(1), [l for l in range(1,self.sh_lmax+1)]),
                            layout=cue.ir_mul,
                            optimize_fallback=True
                    )

      layers = []
      for target_irreps in output_irreps:
        layers.append(Layer(
                      relative_positions_irreps,
                      node_features_irreps,
                      target_irreps))
        node_features_irreps = target_irreps
      
      self.layers = torch.nn.ModuleList(layers)
          
  def forward(self,
              node_features,
              pos,
              edge_index,
              batch):
  
    # Passing in graphs make dynamo angry
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

def build_model():
    model = Model()
    return model

def train(steps=200):
    model = prepare(build_model)()
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=0.01)

    @nvtx.annotate(color="red")
    def loss_fn(model_output, graphs):
        logits = model_output
        labels = graphs.y  # [num_graphs]
        loss = F.cross_entropy(logits, labels)
        loss = torch.mean(loss)
        return loss, logits

    @nvtx.annotate(color="blue")
    def update_fn(model, opt, graphs):
        model_output = model(graphs.numbers, graphs.pos, graphs.edge_index, graphs.batch)
        loss, logits = loss_fn(model_output, graphs)
        
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        labels = graphs.y
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()

        return loss.item(), accuracy.item()

    # Dataset
    graphs = get_tetris()
    # graphs = get_random_graph(n_nodes=50, cutoff=1.5)
    graphs = graphs.to(device=device)

    # Compile
    model = torch.compile(model)

    wall = time.perf_counter()
    print("compiling...", flush=True)
    # Warmup runs
    for i in range(3):
        loss, accuracy = update_fn(model, opt, graphs)
    print(f"initial accuracy = {100 * accuracy:.0f}%", flush=True)
    print(f"compilation took {time.perf_counter() - wall:.1f}s")

    from ctypes import cdll
    libcudart = cdll.LoadLibrary('libcudart.so')

    # Train
    wall = time.perf_counter()
    print("training...", flush=True)
    for step in tqdm(range(steps)):
        if step == 20:
            libcudart.cudaProfilerStart()

        loss, accuracy = update_fn(model, opt, graphs)
        
        if step == 30:
            libcudart.cudaProfilerStop()

        if accuracy == 1.0:
            break

    print(f"final accuracy = {100 * accuracy:.0f}%")
    print(f"training took {time.perf_counter() - wall:.1f}s")
    
    # # Export model
    # so_path = torch._export.aot_compile(
    #             model,
    #             args = (graphs.numbers,graphs.pos,graphs.edge_index,graphs.batch),
    #             options={"aot_inductor.output_path": os.path.join(os.getcwd(), "export/model.so"),
    #         })
    
    # print("node_features", graphs.numbers)
    # print("pos", graphs.pos)
    # print("edge_index", graphs.edge_index)
    # print("batch", graphs.batch)
    
    # runner = torch._C._aoti.AOTIModelContainerRunnerCuda(os.path.join(os.getcwd(), f"export/model.so"), 1, device)
    # outputs_export = runner.run([graphs.numbers,graphs.pos,graphs.edge_index,graphs.batch])
    # print(f"output {outputs_export[0]}")
        

if __name__ == "__main__":
    train()