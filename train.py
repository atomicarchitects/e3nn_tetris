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

# Borrowed from https://github.com/pytorch-labs/gpt-fast/blob/db7b273ab86b75358bd3b014f1f022a19aba4797/generate.py#L16-L18
torch.set_float32_matmul_precision("high")
import torch._dynamo.config
import torch._inductor.config

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
    
from model import Model
from data import get_tetris, get_random_graph

device = "cuda" if torch.cuda.is_available() else "cpu"


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
    graphs = graphs.to(device=device)

    # Compile
    model = torch.compile(model, fullgraph=True, mode="reduce-overhead")

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
    
    # Export model
    so_path = torch._export.aot_compile(
                model,
                args = (graphs.numbers,graphs.pos,graphs.edge_index,graphs.batch),
                options={"aot_inductor.output_path": os.path.join(os.getcwd(), "export/model.so"),
            })
    
    print("Traced Shapes")
    print("node_features", graphs.numbers.shape, graphs.numbers.dtype)
    print("pos", graphs.pos.shape, graphs.pos.dtype)
    print("edge_index", graphs.edge_index.shape, graphs.edge_index.dtype)
    print("batch", graphs.batch.shape, graphs.batch.dtype)

if __name__ == "__main__":
    train()