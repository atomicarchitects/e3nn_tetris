## TODO

- [x] Train tetris
- [x] Get `torch.export` pipeline working
- [ ] Call `model.so` successfully in cpp-land

### (Preliminary) Training time comparision with e3nn-jax on RTX A5500

TODO: Need to test out on bigger datasets/models. The throughput is roughly the same but there might be some initialization differences that make JAX converge faster.

- e3nn + Torch 2

```bash
compiling...
W0815 01:47:51.930000 140183613732416 torch/fx/experimental/symbolic_shapes.py:4449] [0/0] xindex is not in var_ranges, defaulting to unknown range.
initial accuracy = 25%
compilation took 203.1s
training...
 66%|██████████████████████████████████████████████████████████████████▋                                  | 132/200 [00:00<00:00, 448.17it/s]
final accuracy = 100%
training took 0.3s
```

- e3nn-jax

```bash
compiling...
initial accuracy = 25%
compilation took 7.1s
training...
 15%|███████████████▎                                                                                      | 30/200 [00:00<00:00, 473.85it/s]
final accuracy = 100%
training took 0.1s
```
