#!/usr/bin/env python3

import argparse
import struct

import rosbag
from tqdm import tqdm

BINARY_FORMAT = "QHH?"

def process_event_message(ros_msg, start_time=None, end_time=None):
    events = ros_msg.events if hasattr(ros_msg, "events") else [ros_msg]
    extracted_data = []
    for event in events:
        stamp = event.ts if hasattr(event, "ts") else ros_msg.header.stamp
        timestamp_ns = stamp.secs * 1_000_000_000 + stamp.nsecs
        timestamp_sec = timestamp_ns / 1_000_000_000
        if start_time is not None and timestamp_sec < start_time:
            continue
        if end_time is not None and timestamp_sec > end_time:
            continue
        extracted_data.append(
            (
                timestamp_ns,
                getattr(event, "x", 0),
                getattr(event, "y", 0),
                getattr(event, "polarity", False),
            )
        )
    return extracted_data

def convert_rosbag_to_bin(bag_path, topic, output_path, start_time=None, end_time=None):
    with rosbag.Bag(bag_path, "r") as bag:
        total_messages = bag.get_message_count(topic_filters=topic)
        messages = bag.read_messages(topics=[topic])
        with open(output_path, "wb") as output_file:
            for _, ros_msg, _ in tqdm(messages, total=total_messages, desc="Converting"):
                for event_data in process_event_message(ros_msg, start_time, end_time):
                    output_file.write(struct.pack(BINARY_FORMAT, *event_data))
    print(f"Saved converted events to {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Convert a ROS event bag to a binary file.")
    parser.add_argument("bag_path", help="Input ROS bag")
    parser.add_argument("output_path", help="Output binary file")
    parser.add_argument("--topic", default="/dvs/events", help="Event topic")
    parser.add_argument("--start-time", type=float, help="Start time in Unix seconds")
    parser.add_argument("--end-time", type=float, help="End time in Unix seconds")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    convert_rosbag_to_bin(
        args.bag_path,
        args.topic,
        args.output_path,
        args.start_time,
        args.end_time,
    )
