#!/bin/bash

# 检查是否提供了文件名参数
if [ -z "$1" ]; then
  echo "请提供文件名作为参数，例如：./run_rosbag.sh 2025-03-16-14-57-35"
  exit 1
fi

# 设置文件名变量
FILENAME="$1"

# 打开一个新终端并执行命令

gnome-terminal -- bash -c "cd ~/codes/2_Drone_Dection/Dataset_Toolbox; python bag_to_image.py ~/dataset/Added/low_light/${FILENAME}_image.bag ~/dataset/Added/low_light/${FILENAME}; exec bash"

