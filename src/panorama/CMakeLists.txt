cmake_minimum_required(VERSION 3.0.2)
project(panorama)

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
     "Choose the type of build, options are: None(CMAKE_BUILD_TYPE), Debug, Release, RelWithDebInfo, MinSizeRel."
     FORCE)
endif()

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  dvs_msgs
  sensor_msgs
  image_geometry
  image_transport
  cv_bridge
  message_generation
)
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h)
find_library(TENSORRT_LIBRARY nvinfer)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
  message(FATAL_ERROR "TensorRT headers and libraries were not found")
endif()

FILE(GLOB messages_to_build RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/msg"
    "${CMAKE_CURRENT_SOURCE_DIR}/msg/*.msg")

add_message_files(
    FILES
    ${messages_to_build}
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs
)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS scripts/publish_camera_info.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

cuda_add_library(motion_comp_cuda src/frontend/motion_compensation.cu)
cuda_add_library(post_process_cuda src/backend/post_process.cu)

add_executable(Panorama 
    src/panorama.cpp
    src/node.cpp
    src/utils/parameters.cpp
    src/frontend/ang_vel_estimator.cpp 
    src/backend/pose_graph_optimizer.cpp
    src/backend/drone_detector.cpp)

target_compile_options(Panorama PRIVATE -g -O1)

target_link_libraries(Panorama 
    PUBLIC motion_comp_cuda
    PUBLIC post_process_cuda
    PUBLIC OpenMP::OpenMP_CXX
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${cv_bridge_LIBRARIES}
    -lglog
    gflags
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBRARY}
)
add_dependencies(Panorama ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
