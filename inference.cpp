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
        torch::randn({32,1}, at::kCUDA),
        torch::randn({32,3}, at::kCUDA),
        torch::randn({2,50}, at::kCUDA),
        torch::randn({32,1}, at::kCUDA)
    };
    std::vector<torch::Tensor> outputs = runner->run(inputs);
    std::cout << "Result from the first inference:"<< std::endl;
    std::cout << outputs[0] << std::endl;
    return 0;
}