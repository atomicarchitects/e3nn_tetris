cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(aoti_example)

find_package(CUDA)

find_package(Torch REQUIRED)

add_executable(inference inference.cpp)

target_link_libraries(inference "${TORCH_LIBRARIES}" m)
set_property(TARGET inference PROPERTY CXX_STANDARD 17)