cmake_minimum_required(VERSION 3.10)
project(infer_tensorrt)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

# 设置 TensorRT 与 CUDA 库路径（Orin NX 默认安装在系统路径）
find_library(TRT_LIB nvinfer)
find_library(CUDART_LIB cudart)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    /usr/include     # TRT headers usually here
)

add_executable(infer_tensorrt_temporal infer_tensorrt_temporal.cpp)
target_link_libraries(infer_tensorrt_temporal
    ${OpenCV_LIBS}
    ${TRT_LIB}
    ${CUDART_LIB}
)
