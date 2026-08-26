cmake_minimum_required(VERSION 3.0.2)
project(datasync)

set(CMAKE_BUILD_TYPE Debug)
add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  dvs_msgs
  message_filters
  roscpp
  rospy
  rosbag
  std_msgs
  sensor_msgs
  image_transport
  cv_bridge
  geometry_msgs
  visualization_msgs
)

find_package(OpenCV REQUIRED)
find_package(OpenCL)
find_package(Eigen3 REQUIRED)

catkin_package()

if (NOT OPENCL_FOUND)
    set(OpenCL_LIBRARY "")
    set(OpenCL_INCLUDE_DIRS "")
endif (NOT OPENCL_FOUND)

add_definitions(-DOPENCL_ENABLED=${OPENCL_FOUND})
include_directories(
    .
    include
    ${EIGEN3_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
    ${OpenCL_INCLUDE_DIRS}
    ${Opencv_INCLUDE_DIRS}
)

add_executable(Motion_Compensation_Event src/Motion_Compensation_Event.cpp)
target_link_libraries(Motion_Compensation_Event 
			${catkin_LIBRARIES}
			${OpenCV_LIBS}
	        ${OpenCL_LIBRARY}
)
add_dependencies(Motion_Compensation_Event ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(IMU_Motion_Compensation src/IMU_Motion_Compensation.cpp)
target_link_libraries(IMU_Motion_Compensation 
			${catkin_LIBRARIES}
			${OpenCV_LIBS}
	        ${OpenCL_LIBRARY}
)
add_dependencies(IMU_Motion_Compensation ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(Initial_Motion_Compensation src/Initial_Motion_Compensation.cpp)
target_link_libraries(Initial_Motion_Compensation 
			${catkin_LIBRARIES}
			${OpenCV_LIBS}
	        ${OpenCL_LIBRARY}
)
add_dependencies(Initial_Motion_Compensation ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
