> 这部分代码用于接收来自USB的光电门开关信号

## 准备工作

查看USB设备：`ls /dev/ttyUSB*`

设置权限：`sudo chmod 666 /dev/ttyUSB0`

## 运行

运行命令： `roslaunch photogate_reader photo_gate.launch`

光电门开关的Topic：`/photogate_reader/photogate_state`