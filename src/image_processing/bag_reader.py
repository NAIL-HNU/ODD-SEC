#!/usr/bin/env python3

import argparse
import csv

import rosbag

def read_frame_info(bag_path, topic):
    frame_info = []
    with rosbag.Bag(bag_path, "r") as bag:
        for _, message, _ in bag.read_messages(topics=[topic]):
            if hasattr(message, "header"):
                stamp = message.header.stamp
            elif getattr(message, "events", None):
                stamp = message.events[0].ts
            else:
                stamp = None
            timestamp = None if stamp is None else stamp.secs + stamp.nsecs * 1e-9
            event_count = len(message.events) if hasattr(message, "events") else 1
            frame_info.append((timestamp, event_count))
    return frame_info

def main():
    parser = argparse.ArgumentParser(description="Inspect event counts in a ROS bag.")
    parser.add_argument("bag_path", help="Input ROS bag")
    parser.add_argument("--topic", default="/dvs/events", help="Event topic")
    parser.add_argument("--csv", dest="csv_path", help="Optional CSV output path")
    args = parser.parse_args()

    frame_info = read_frame_info(args.bag_path, args.topic)
    for index, (timestamp, event_count) in enumerate(frame_info):
        timestamp_text = "n/a" if timestamp is None else f"{timestamp:.9f}"
        print(f"Frame {index}: header_time={timestamp_text}, event_count={event_count}")

    if args.csv_path:
        with open(args.csv_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Frame", "HeaderTime", "EventCount"])
            for index, (timestamp, event_count) in enumerate(frame_info):
                writer.writerow([index, timestamp, event_count])

if __name__ == "__main__":
    main()
