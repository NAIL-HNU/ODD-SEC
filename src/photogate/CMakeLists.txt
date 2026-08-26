cmake_minimum_required(VERSION 3.0.2)
project(photogate_reader)

set(CMAKE_CXX_STANDARD 14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  serial
)

catkin_package(
  CATKIN_DEPENDS roscpp std_msgs serial
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(photogate_node src/photogate_node.cpp)

target_link_libraries(photogate_node
  ${catkin_LIBRARIES}
)