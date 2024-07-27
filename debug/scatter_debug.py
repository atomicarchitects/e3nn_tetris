import torch

torch._dynamo.config.capture_scalar_outputs = True

def my_arithmetic(a, b):
    wrk = torch.zeros(a.size(0), dtype=torch.float32)
    wrk.scatter_add_(0, b, torch.ones_like(b, dtype=torch.float32))
    return wrk

model = torch.compile(my_arithmetic, fullgraph=True, disable=True)
my_a = torch.randn([9])
my_b = torch.ones(9, dtype=torch.int64)
print(model(my_a, my_b))