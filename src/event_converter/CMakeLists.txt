cmake_minimum_required(VERSION 3.0.2)
project(event_converter)

find_package(catkin REQUIRED COMPONENTS
  dvs_msgs
  rosbag
  roscpp
  visualization_msgs
  image_transport
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(event_bag_to_dat src/event_bag_to_dat.cpp)

target_link_libraries(event_bag_to_dat
  ${catkin_LIBRARIES}
)
