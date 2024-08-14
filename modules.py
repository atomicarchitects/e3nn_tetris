from typing import List, Union, Optional, Callable

import torch
import torch.nn.functional as F
import numpy as np
from e3nn import o3

class _MulIndexSliceHelper:
    irreps: o3.Irreps

    def __init__(self, irreps) -> None:
        self.irreps = irreps

    def __getitem__(self, index: slice) -> o3.Irreps:
        if not isinstance(index, slice):
            raise IndexError("Irreps.slice_by_mul only supports slices.")

        start, stop, stride = index.indices(self.irreps.num_irreps)
        if stride != 1:
            raise NotImplementedError("Irreps.slice_by_mul does not support strides.")

        out = []
        i = 0
        for mul, ir in self.irreps:
            if start <= i and i + mul <= stop:
                out.append((mul, ir))
            elif start < i + mul and i < stop:
                out.append((min(stop, i + mul) - max(start, i), ir))
            i += mul
        return o3.Irreps(out)

    
def slice_by_mul(irreps):
    return _MulIndexSliceHelper(irreps)

def filter(
    irreps,
    keep: Union["o3.Irreps", List[o3.Irrep]] = None,
    *,
    drop: Union["o3.Irreps", List[o3.Irrep]] = None,
    lmax: int = None,
) -> "o3.Irreps":
    if keep is None and drop is None and lmax is None:
        return self
    if keep is not None and drop is not None:
        raise ValueError("Cannot specify both keep and drop")
    if keep is not None and lmax is not None:
        raise ValueError("Cannot specify both keep and lmax")
    if drop is not None and lmax is not None:
        raise ValueError("Cannot specify both drop and lmax")

    if keep is not None:
        if isinstance(keep, str):
            keep = o3.Irreps(keep)
        if isinstance(keep, o3.Irrep):
            keep = [keep]
        keep = {o3.Irrep(ir) for ir in keep}
        return o3.Irreps([(mul, ir) for mul, ir in irreps if ir in keep])

    if drop is not None:
        if isinstance(drop, str):
            drop = o3.Irreps(drop)
        if isinstance(drop, o3.Irrep):
            drop = [drop]
        drop = {o3.Irrep(ir) for ir in drop}
        return o3.Irreps([(mul, ir) for mul, ir in irreps if ir not in drop])

    if lmax is not None:
        return o3.Irreps([(mul, ir) for mul, ir in irreps if ir.l <= lmax])

def soft_odd(x):
    return (1 - torch.exp(-(x**2))) * x

def normalspace(n: int) -> torch.Tensor:
    return np.sqrt(2) * torch.erfinv(torch.linspace(-1.0, 1.0, n + 2)[1:-1])


def normalize_function(phi: Callable[[float], float]) -> Callable[[float], float]:    
    x = normalspace(1_000_001)
    c = torch.mean(phi(x) ** 2) ** 0.5
    c = c.item()

    if np.allclose(c, 1.0):
        return phi
    else:

        def rho(x):
            return phi(x) / c

        return rho

def parity_function(phi: Callable[[float], float]) -> int:
    x = torch.linspace(0.0, 10.0, 256)

    a1, a2 = phi(x), phi(-x)
    if torch.max(torch.abs(a1 - a2)) < 1e-5:
        return 1
    elif torch.max(torch.abs(a1 + a2)) < 1e-5:
        return -1
    else:
        return 0

def is_zero_in_zero(phi: Callable[[float], float]) -> bool:
    return torch.allclose(phi(torch.Tensor([0.0])), 0.0)

# class ScalarActivation(nn.Module):
    
#     def __init__(self, 
#                  irreps_in: o3.Irreps,
#                  acts: List[Optional[Callable[[float], float]]] = None,
#                  *,
#                  even_act: Callable[[float], float] = F.gelu,
#                  odd_act: Callable[[float], float] = soft_odd,
#                  normalize_act: bool = True):

#         super(ScalarActivation, self).__init__()

#         if acts is None:
#             acts = [
#                 {1: even_act, -1: odd_act}[ir.p] if ir.l == 0 else None
#                 for _, ir in irreps_in
#             ]

#         assert len(irreps_in) == len(acts), (irreps_in, acts)
#         irreps_out = []
#         paths = {}

#         for (mul, (l_in, p_in)), slice_x, act in zip(irreps_in, irreps_in.slices(), acts):
#             if act is not None:
#                 if l_in != 0:
#                     raise ValueError(
#                         f"Activation: cannot apply an activation function to a non-scalar input. {irreps_in} {acts}"
#                     )

#                 if normalize_act:
#                     act = normalize_function(act)

#                 p_out = parity_function(act) if p_in == -1 else p_in
#                 if p_out == 0:
#                     raise ValueError(
#                         "Activation: the parity is violated! The input scalar is odd but the activation is neither even nor odd."
#                     )

#                 irreps_out.append((mul, (0, p_out)))
#             else:
#                 irreps_out.append((mul, (l_in, p_in)))
                
#             paths[l_in] = (slice_x, act)

#         self._same_acts = False
#         # for performance, if all the activation functions are the same, we can apply it to the contiguous array as well:
#         if acts and acts.count(acts[0]) == len(acts):
#             if acts[0] is None:
#                 self.act = None
#             else:
#                 act = acts[0]
#                 if normalize_act:
#                     self.act = normalize_function(act)
 
#         irreps_out = o3.Irreps(irreps_out)
#         self.irreps_out, _, self.inv = irreps_out.sort()
#         self.paths = paths

#     def forward(self, input: torch.Tensor):
        
#         if self._same_acts:
#             if self.act is None:
#                 return input
#             else:
#                 return self.act(input)
    
#         chunks = []
#         for (slice_x, act) in self.paths.values():
#             if act is None:
#                 chunks.append(input[..., slice_x])
#             else:
#                 chunks.append(act(input[..., slice_x]))

#         return torch.cat([chunks[i] for i in self.inv], dim=-1)

# class Gate(torch.nn.Module):
#   def __init__(
#         self,
#         irreps: o3.Irreps,
#         even_act: Callable[[float], float] = F.gelu,
#         odd_act: Callable[[float], float] = soft_odd,
#         even_gate_act: Callable[[float], float] = F.sigmoid,
#         odd_gate_act: Callable[[float], float] = F.tanh,
#         normalize_act: bool = True):
      
#         scalars_irreps = filter(irreps, keep=["0e", "0o"])
#         vectors_irreps = filter(irreps, drop=["0e", "0o"])
        
#         if scalars_irreps.dim < vectors_irreps.num_irreps:
#             raise ValueError(
#                 "The input must have at least as many scalars as the number of non-scalar irreps"
#             )
#         scalars_extra_irreps = scalars_irreps.slice_by_mul[
#             : scalars_irreps.irreps.dim - vectors_irreps.irreps.num_irreps
#         ]
#         scalars_gates_irreps = scalars_irreps.slice_by_mul[
#             scalars_irreps.irreps.dim - vectors_irreps.irreps.num_irreps :
#         ]
        
#         self.scalars_extra = ScalarActivation(
#             scalars_extra_irreps,
#             even_act=even_act,
#             odd_act=odd_act,
#             normalize_act=normalize_act
#         )
#         self.scalars_gates = ScalarActivation(
#             scalars_gates_irreps,
#             even_act=even_gate_act,
#             odd_act=odd_gate_act,
#             normalize_act=normalize_act,
#         )
        
#         self.elementwise_tp = o3.ElementwiseTensorProduct(scalars_extra_irreps, vectors_irreps)
        

#         self.output_irreps = self.scalars_extra_irreps + self.elementwise_tp.irreps_out

    
    
