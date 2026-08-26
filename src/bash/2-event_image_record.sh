#!/bin/bash

# 检查是否提供了文件名参数
if [ -z "$1" ]; then
  echo "请提供文件名作为参数，例如：./run_rosbag.sh 2025-03-16-14-57-35"
  exit 1
fi

# 设置文件名变量
FILENAME="$1"

# 打开一个新终端并执行命令

gnome-terminal -- bash -c "cd ~/dataset/Added/low_light; rosbag record /count_image -O ${FILENAME}_image.bag; exec bash"

# gnome-terminal -- bash -c "cd ~/dataset/Added/low_light; rosbag record /event_new -O ${FILENAME}_event.bag; exec bash"
sleep 1
gnome-terminal -- bash -c " rosbag play -r 0.1 ${FILENAME}.bag; exec bash"
