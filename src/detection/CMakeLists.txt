cmake_minimum_required(VERSION 3.10)
project(infer_tensorrt)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

find_library(TRT_LIB nvinfer)
find_path(TRT_INCLUDE_DIR NvInfer.h)

if(NOT TRT_LIB OR NOT TRT_INCLUDE_DIR)
    message(FATAL_ERROR "TensorRT and CUDA development files were not found")
endif()

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${TRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

add_executable(infer_tensorrt_temporal infer_tensorrt_temporal.cpp)
target_link_libraries(infer_tensorrt_temporal
    ${OpenCV_LIBS}
    ${TRT_LIB}
    ${CUDA_CUDART_LIBRARY}
)
