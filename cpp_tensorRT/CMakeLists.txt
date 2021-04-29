cmake_minimum_required(VERSION 3.19)

#设置CUDA编译器，据自己的需要更改路径
SET(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(cpp_tensorRT C CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)


include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

add_definitions(-O2 -pthread)

file(GLOB Project_DIR "*.h" "*.c" "*.cc" "*.cu")

add_executable(cpp_tensorRT ${Project_DIR})

set_target_properties(
    cpp_tensorRT
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

#find_package(COMPONENTS REQUIRED)
target_link_libraries(${PROJECT_NAME}  ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} nvinfer)
target_link_libraries(${PROJECT_NAME} cudart)