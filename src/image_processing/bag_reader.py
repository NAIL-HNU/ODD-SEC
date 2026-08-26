import rosbag

BAG_PATH = '/home/zhx/dataset/data/2025-07-04/2025-07-04-19-44-46_static.bag'  # 替换为你的bag路径
EVENT_TOPIC = '/dvs/events'                      # 替换为你的事件topic

frame_infos = []

with rosbag.Bag(BAG_PATH, 'r') as bag:
    for _, ros_msg, _ in bag.read_messages(topics=[EVENT_TOPIC]):
        # 获取header时间戳（秒+纳秒）
        if hasattr(ros_msg, 'header'):
            header_stamp = ros_msg.header.stamp
            header_time = header_stamp.secs + header_stamp.nsecs * 1e-9
        elif hasattr(ros_msg, 'events') and len(ros_msg.events) > 0 and hasattr(ros_msg.events[0], 'ts'):
            # 某些消息没有header，取第一个事件的ts
            ts = ros_msg.events[0].ts
            header_time = ts.secs + ts.nsecs * 1e-9
        else:
            header_time = None

        # 统计事件数
        if hasattr(ros_msg, 'events'):
            count = len(ros_msg.events)
        else:
            count = 1  # 单事件消息

        frame_infos.append((header_time, count))

# 打印结果
for idx, (header_time, count) in enumerate(frame_infos):
    print(f"Frame {idx}: header_time={header_time:.9f}, event_count={count}")

# 可选：保存为CSV
import csv
with open('event_frame_info.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Frame', 'HeaderTime', 'EventCount'])
    for idx, (header_time, count) in enumerate(frame_infos):
        writer.writerow([idx, header_time, count])
