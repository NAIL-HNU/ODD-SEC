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
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
# find_package(OpenCL)
# find_package(PCL)

# search for all msg files
FILE(GLOB messages_to_build RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/msg"
    "${CMAKE_CURRENT_SOURCE_DIR}/msg/*.msg")

    
# message(WARNING "Found .msg files: ${messages_to_build}")

add_message_files(
    FILES
    ${messages_to_build}  # 确保这里使用的是上面 FILE(GLOB) 设置的变量
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs
)

include_directories(/usr/include/eigen3)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add CUDA include directories (adjust if necessary)
include_directories(/usr/local/cuda/include) # Default CUDA include path
include_directories(/usr/src/tensorrt/include) # Default TensorRT include path, adjust if your version/path is different

# Add CUDA library directories (adjust if necessary)
link_directories(/usr/local/cuda/lib64) # Default CUDA library path
link_directories(/usr/lib/aarch64-linux-gnu) # Common location for TensorRT libraries on Jetson
link_directories(/usr/src/tensorrt/lib) # Default TensorRT library path, adjust if your version/path is different

# catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES panorama
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
# )
# if (NOT OPENCL_FOUND)
#     set(OpenCL_LIBRARY "")
#     set(OpenCL_INCLUDE_DIRS "")
# endif (NOT OPENCL_FOUND)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr")

# add_definitions(-DOPENCL_ENABLED=${OPENCL_FOUND})

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include  # 添加项目自身的include目录
    ${catkin_INCLUDE_DIRS}
		${OpenCV_LIBS}
	  # ${OpenCL_LIBRARY}
    ${catkin_INCLUDE_DIRS}
    /usr/include/glog
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
    # image_transport
    -lglog
    gflags
    cudart
    nvinfer
    nvparsers
    nvonnxparser
    nvToolsExt
)
add_dependencies(Panorama ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})