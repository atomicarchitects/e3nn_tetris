## e3nn_tetris

Depends on https://github.com/e3nn/e3nn/pull/455


### Train

```python
(.venv) mkotak@radish:~/atomic_architects/projects/e3nn_tetris$ python train.py 
compiling...
W0815 21:09:54.341000 140232380905024 torch/fx/experimental/symbolic_shapes.py:4449] [0/0] xindex is not in var_ranges, defaulting to unknown range.
initial accuracy = 12%
compilation took 20.7s
training...
 34%|██████████████████████████████████▋                                                                   | 68/200 [00:00<00:00, 651.29it/s]
final accuracy = 100%
training took 0.1s
W0815 21:10:06.125000 140242116276224 torch/fx/experimental/symbolic_shapes.py:4449] rindex is not in var_ranges, defaulting to unknown range.
node_features tensor([[1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.]], device='cuda:0')
pos tensor([[0., 0., 0.],
        [0., 0., 1.],
        [1., 0., 0.],
        [1., 1., 0.],
        [1., 1., 1.],
        [1., 1., 2.],
        [2., 1., 1.],
        [2., 0., 1.],
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [1., 1., 0.],
        [0., 0., 0.],
        [0., 0., 1.],
        [0., 0., 2.],
        [0., 0., 3.],
        [0., 0., 0.],
        [0., 0., 1.],
        [0., 1., 0.],
        [1., 0., 0.],
        [0., 0., 0.],
        [0., 0., 1.],
        [0., 0., 2.],
        [0., 1., 0.],
        [0., 0., 0.],
        [0., 0., 1.],
        [0., 0., 2.],
        [0., 1., 1.],
        [0., 0., 0.],
        [1., 0., 0.],
        [1., 1., 0.],
        [2., 1., 0.]], device='cuda:0')
edge_index tensor([[ 1,  2,  0,  0,  3,  2,  5,  6,  4,  4,  7,  6,  9, 10,  8, 11,  8, 11,
          9, 10, 13, 12, 14, 13, 15, 14, 17, 18, 19, 16, 16, 16, 21, 23, 20, 22,
         21, 20, 25, 24, 26, 27, 25, 25, 29, 28, 30, 29, 31, 30],
        [ 0,  0,  1,  2,  2,  3,  4,  4,  5,  6,  6,  7,  8,  8,  9,  9, 10, 10,
         11, 11, 12, 13, 13, 14, 14, 15, 16, 16, 16, 17, 18, 19, 20, 20, 21, 21,
         22, 23, 24, 25, 25, 25, 26, 27, 28, 29, 29, 30, 30, 31]],
       device='cuda:0')
batch tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5,
        6, 6, 6, 6, 7, 7, 7, 7], device='cuda:0')
output tensor([[ 0.0344, -0.0344, -0.1076, -0.4316, -0.1765, -0.1971, -0.2294, -0.3128],
        [-0.0559,  0.0559,  0.0109, -0.1000, -0.3582, -0.6548, -0.4990, -0.4543],
        [-0.1819,  0.1819,  0.8137, -0.0546, -0.5677, -0.0819, -0.5391,  0.4103],
        [ 0.0096, -0.0096,  0.3495,  0.4046, -0.3325, -0.0950, -0.0316, -0.3742],
        [ 0.0054, -0.0054, -0.1696, -0.5817,  0.6609,  0.3416,  0.0284, -0.6731],
        [ 0.0231, -0.0231,  0.1962, -0.0599, -0.0899,  0.4756,  0.2846, -0.3791],
        [ 0.0878, -0.0878,  0.2029, -0.0223, -0.0784,  0.3471,  0.5549, -0.4884],
        [-0.1149,  0.1149,  0.4429, -0.1543, -0.4349, -0.0615, -0.5707,  0.7809]],
       device='cuda:0')
```



### Inference


```python
(.venv) mkotak@radish:~/atomic_architects/projects/e3nn_tetris$ make run
./build/inference /home/mkotak/atomic_architects/projects/e3nn_tetris/export/model.so
output tensor 0.0344 -0.0344 -0.1076 -0.4316 -0.1765 -0.1971 -0.2294 -0.3128
-0.0559  0.0559  0.0109 -0.1000 -0.3582 -0.6548 -0.4990 -0.4543
-0.1819  0.1819  0.8137 -0.0546 -0.5677 -0.0819 -0.5391  0.4103
 0.0096 -0.0096  0.3495  0.4046 -0.3325 -0.0950 -0.0316 -0.3742
 0.0054 -0.0054 -0.1696 -0.5817  0.6609  0.3416  0.0284 -0.6731
 0.0231 -0.0231  0.1962 -0.0599 -0.0899  0.4756  0.2846 -0.3791
 0.0878 -0.0878  0.2029 -0.0223 -0.0784  0.3471  0.5549 -0.4884
-0.1149  0.1149  0.4429 -0.1543 -0.4349 -0.0615 -0.5707  0.7809
[ CUDAFloatType{8,8} ]
```




### (Preliminary) Training time comparision with e3nn-jax on RTX A5500

TODO: Need to test out on bigger datasets/models. The throughput is roughly the same but there might be some initialization differences that make JAX converge faster.

- e3nn + Torch 2

```python
compiling...
initial accuracy = 12%
compilation took 127.5s
training...
 34%|███████████████████████████████████                                                                    | 68/200 [00:00<00:00, 797.43it/s]
final accuracy = 100%
training took 0.1s
```

- e3nn-jax

```python
compiling...
initial accuracy = 12%
compilation took 5.3s
training...
 12%|███████████▋                                                                                     | 24/200 [00:00<00:00, 436.59it/s]
final accuracy = 100%
```
