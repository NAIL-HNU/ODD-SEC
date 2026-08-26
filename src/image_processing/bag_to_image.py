#!/usr/bin/env python3
import rosbag
import cv2
from cv_bridge import CvBridge
import os
import argparse

parser = argparse.ArgumentParser(description='Extract images from a ROS bag file.')
parser.add_argument('bag_file', type=str, help='Path to the input bag file')
parser.add_argument('output_folder', type=str, help='Path to the output folder for images')
args = parser.parse_args()

bag_file = args.bag_file
output_folder = args.output_folder

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

bridge = CvBridge()

bag = rosbag.Bag(bag_file, 'r')
first_event_timestamp = 0
first_event_received = False
t0=10

for topic, msg, _ in bag.read_messages():
    if topic == '/count_image':
        if first_event_received == False:
            first_event_received=True
            first_event_timestamp=(msg.header.stamp.secs * 1e6) + (msg.header.stamp.nsecs / 1e3)
         
        try:
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            

            timestamp = (msg.header.stamp.secs * 1e6) + (msg.header.stamp.nsecs / 1e3)
            
            
            filename = os.path.join(output_folder, f'{timestamp}.jpg')

            
            cv2.imwrite(filename, cv_image)
            print(f'Saved {filename}')
        except Exception as e:
            print(f'Error processing image: {e}')

bag.close()
