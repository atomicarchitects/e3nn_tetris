#include <iostream>
#include <vector>

#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>

int main(int argc, char *argv[]) {
    
    char *model_path = NULL;  // e.g. mode.so
    model_path = argv[1];

    c10::InferenceMode mode;
    torch::inductor::AOTIModelContainerRunnerCuda *runner;
    runner = new torch::inductor::AOTIModelContainerRunnerCuda(model_path, 1);
    std::vector<torch::Tensor> inputs = {
        // node_features

        torch::ones({32, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)),


        // pos
        torch::tensor({
            {0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, {1., 1., 0.},
            {1., 1., 1.}, {1., 1., 2.}, {2., 1., 1.}, {2., 0., 1.},
            {0., 0., 0.}, {1., 0., 0.}, {0., 1., 0.}, {1., 1., 0.},
            {0., 0., 0.}, {0., 0., 1.}, {0., 0., 2.}, {0., 0., 3.},
            {0., 0., 0.}, {0., 0., 1.}, {0., 1., 0.}, {1., 0., 0.},
            {0., 0., 0.}, {0., 0., 1.}, {0., 0., 2.}, {0., 1., 0.},
            {0., 0., 0.}, {0., 0., 1.}, {0., 0., 2.}, {0., 1., 1.},
            {0., 0., 0.}, {1., 0., 0.}, {1., 1., 0.}, {2., 1., 0.}
        }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)),

        // edge_index
        torch::tensor({
        { 1,  2,  0,  0,  3,  2,  5,  6,  4,  4,  7,  6,  9, 10,  8, 11,  8, 11,
          9, 10, 13, 12, 14, 13, 15, 14, 17, 18, 19, 16, 16, 16, 21, 23, 20, 22,
         21, 20, 25, 24, 26, 27, 25, 25, 29, 28, 30, 29, 31, 30},
        { 0,  0,  1,  2,  2,  3,  4,  4,  5,  6,  6,  7,  8,  8,  9,  9, 10, 10,
         11, 11, 12, 13, 13, 14, 14, 15, 16, 16, 16, 17, 18, 19, 20, 20, 21, 21,
         22, 23, 24, 25, 25, 25, 26, 27, 28, 29, 29, 30, 30, 31}
        }, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA)),

        // batch
        torch::tensor({
        0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5,
        6, 6, 6, 6, 7, 7, 7, 7
        }, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA))
    };
    std::vector<torch::Tensor> outputs = runner->run(inputs);
    std::cout << "output tensor" << outputs[0] << std::endl;
    return 0;
}