import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm.auto import tqdm
from e3nn import o3

# Borrowed from https://github.com/pytorch-labs/gpt-fast/blob/db7b273ab86b75358bd3b014f1f022a19aba4797/generate.py#L16-L18
torch.set_float32_matmul_precision("high")
import torch._dynamo.config
import torch._inductor.config

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
    
from model import Model
from data import get_tetris, get_random_graph

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(steps=200):
    model = Model()
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=0.01)

    def loss_fn(model_output, graphs):
        logits = model_output
        labels = graphs.y  # [num_graphs]
        loss = F.cross_entropy(logits, labels)
        loss = torch.mean(loss)
        return loss, logits

    def update_fn(model, opt, graphs):
        model_output = model(graphs)
        loss, logits = loss_fn(model_output, graphs)
        
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        labels = graphs.y
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()

        return loss.item(), accuracy.item()

    # Compile the update function
    update_fn_compiled = torch.compile(update_fn, mode="reduce-overhead")

    # Dataset
    graphs = get_tetris()
    graphs = graphs.to(device=device)

    # compile jit
    wall = time.perf_counter()
    print("compiling...", flush=True)
    # Warmup runs
    for i in range(3):
        loss, accuracy = update_fn_compiled(model, opt, graphs)
    print(f"initial accuracy = {100 * accuracy:.0f}%", flush=True)
    print(f"compilation took {time.perf_counter() - wall:.1f}s")

    # Train
    wall = time.perf_counter()
    print("training...", flush=True)
    for _ in tqdm(range(steps)):
        loss, accuracy = update_fn_compiled(model, opt, graphs)

        if accuracy == 1.0:
            break

    print(f"final accuracy = {100 * accuracy:.0f}%")
    print(f"training took {time.perf_counter() - wall:.1f}s")

if __name__ == "__main__":
    train()