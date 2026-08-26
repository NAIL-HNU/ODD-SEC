#!/bin/bash

# 获取所有活跃ROS节点名称
nodes=$(rosnode list | grep -v "/rosout")  # 排除rosout

# 提取每个节点的PID并监控
pids=()
for node in $nodes; do
    pids+=($(rosnode info $node 2>/dev/null | grep Pid | awk '{print $2}'))
done

top -p $(IFS=,; echo "${pids[*]}") -b -d 2 | tee ros_nodes_top.txt