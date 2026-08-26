cmake_minimum_required(VERSION 3.0.2)
project(photogate_reader)

## 编译选项
set(CMAKE_CXX_STANDARD 14)

## 查找catkin依赖
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  serial
)

## 声明catkin包
catkin_package(
  CATKIN_DEPENDS roscpp std_msgs serial
)

## 包含头文件目录
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## 添加可执行文件
add_executable(photogate_node src/photogate_node.cpp)

## 链接库
target_link_libraries(photogate_node
  ${catkin_LIBRARIES}
)