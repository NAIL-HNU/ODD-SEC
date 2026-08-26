#!/bin/bash

nodes=$(rosnode list | grep -v "/rosout")

pids=()
for node in $nodes; do
    pids+=($(rosnode info $node 2>/dev/null | grep Pid | awk '{print $2}'))
done

top -p $(IFS=,; echo "${pids[*]}") -b -d 2 | tee ros_nodes_top.txt