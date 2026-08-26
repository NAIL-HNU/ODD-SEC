#!/usr/bin/env /usr/bin/python3
# -*- coding: utf-8 -*-
import rosbag
import cv2
from cv_bridge import CvBridge
import os
import argparse

# 设置命令行参数
parser = argparse.ArgumentParser(description='Extract images from a ROS bag file.')
parser.add_argument('bag_file', type=str, help='Path to the input bag file')
parser.add_argument('output_folder', type=str, help='Path to the output folder for images')
args = parser.parse_args()

# 获取命令行参数
bag_file = args.bag_file
output_folder = args.output_folder

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 初始化CvBridge
bridge = CvBridge()

# 打开bag文件
bag = rosbag.Bag(bag_file, 'r')
first_event_timestamp = 0
first_event_received = False
t0=10

# 遍历bag文件中的消息
for topic, msg, _ in bag.read_messages():
    # 检查消息类型是否为图像
    if topic == '/count_image':
        if first_event_received == False:
            first_event_received=True
            first_event_timestamp=(msg.header.stamp.secs * 1e6) + (msg.header.stamp.nsecs / 1e3)
         
        try:
            # 将ROS图像消息转换为OpenCV图像
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # 生成文件名

            timestamp = (msg.header.stamp.secs * 1e6) + (msg.header.stamp.nsecs / 1e3)
            
            # timestamp_str = str(int(timestamp)).zfill(8)
            
            # 生成文件名
            filename = os.path.join(output_folder, f'{timestamp}.jpg')

            
            # 保存图像为JPG格式
            cv2.imwrite(filename, cv_image)
            print(f'Saved {filename}')
        except Exception as e:
            print(f'Error processing image: {e}')

# 关闭bag文件
bag.close()