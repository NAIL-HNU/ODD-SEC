#!/bin/bash

# 打开一个新终端并执行命令
gnome-terminal -- bash -c "cd /home/dai/workspace/Dataset_Toolbox; source devel/setup.bash;sleep 1; roslaunch datasync Motion_Compensation_Event.launch; exec bash"
