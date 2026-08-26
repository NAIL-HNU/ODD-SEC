# --- 配置参数 ---
ROS_BAG_PATH = '/home/zhx/result/result_events.bag'  # 替换为你的 ROSbag 文件路径
EVENT_TOPIC = '/dvs/events'  # 替换为你的事件流话题名称
OUTPUT_BIN_PATH = '/home/zhx/result/events_1.bin'     # 输出的 .bin 文件路径

import rosbag # 用于 ROS1 的 bag 文件
# from rosbags.highlevel import AnyReader # 用于 ROS2 的 bag 文件
# from pathlib import Path # 用于 ROS2 的 bag 文件路径

import struct # 用于将数据打包成二进制格式
from tqdm import tqdm # 导入 tqdm 库

# --- 定义你的二进制数据格式 ---
# 这是一个示例，假设每个事件包含：
# - timestamp_ns: uint64 (纳秒时间戳) -> 'Q'
# - x: uint16             -> 'H'
# - y: uint16             -> 'H'
# - polarity: bool        -> '?' (等同于 1 字节的 bool)
BINARY_FORMAT = 'QHH?' # 请根据你的实际事件数据结构调整！

def process_event_message(ros_msg, start_time=None, end_time=None):
    """
    从 ROS 事件消息中提取你需要的数据，并根据事件时间戳筛选区间。
    """
    extracted_data = []

    # 如果消息包含一个 'events' 列表
    if hasattr(ros_msg, 'events'):
        for event in ros_msg.events:
            timestamp_ns = event.ts.secs * 1_000_000_000 + event.ts.nsecs if hasattr(event, 'ts') else 0
            timestamp_sec = timestamp_ns / 1_000_000_000
            # tqdm.write(f"timestamp_sec: {timestamp_sec}")
            if (start_time is not None and timestamp_sec < start_time) or (end_time is not None and timestamp_sec > end_time):
                continue
            x = event.x if hasattr(event, 'x') else 0
            y = event.y if hasattr(event, 'y') else 0
            polarity = event.polarity if hasattr(event, 'polarity') else False
            extracted_data.append((timestamp_ns, x, y, polarity))
    else:
        timestamp_ns = ros_msg.header.stamp.secs * 1_000_000_000 + ros_msg.header.stamp.nsecs if hasattr(ros_msg, 'header') else 0
        timestamp_sec = timestamp_ns / 1_000_000_000
        if (start_time is not None and timestamp_sec < start_time) or (end_time is not None and timestamp_sec > end_time):
            return []
        x = ros_msg.x if hasattr(ros_msg, 'x') else 0
        y = ros_msg.y if hasattr(ros_msg, 'y') else 0
        polarity = ros_msg.polarity if hasattr(ros_msg, 'polarity') else False
        extracted_data.append((timestamp_ns, x, y, polarity))
    return extracted_data


def convert_rosbag_to_bin(bag_path, topic, bin_path, process_func, bin_format, start_time=None, end_time=None):
    print(f"正在处理 ROSbag: {bag_path}")
    print(f"正在提取话题: {topic}")

    try:
        # --- 处理 ROS1 Bag 文件 ---
        with rosbag.Bag(bag_path, 'r') as bag:
            # 获取总消息数量以便 tqdm 显示进度
            total_messages = bag.get_message_count(topic_filters=topic)

            with open(bin_path, 'wb') as bin_file:
                # 使用 tqdm 包装 bag.read_messages()，添加进度条
                for current_topic, ros_msg, t in tqdm(bag.read_messages(topics=[topic]), total=total_messages, desc="转换进度"):
                    # 直接处理所有消息，筛选逻辑在process_event_message中
                    events_data_list = process_func(ros_msg, start_time, end_time)
                    for event_data in events_data_list:
                        packed_data = struct.pack(bin_format, *event_data)
                        bin_file.write(packed_data)
        print(f"成功将事件转换并保存到 {bin_path}")

        # --- 处理 ROS2 Bag 文件 (概念性代码，需要根据 rosbags 库的 API 调整) ---
        # from rosbags.highlevel import AnyReader
        # with AnyReader([Path(bag_path)]) as reader:
        #     # 获取总消息数量（ROS2 rosbags 库获取总数可能略有不同，需查阅其文档）
        #     # For example: total_messages = sum(1 for conn, _, _ in reader.messages() if conn.topic == topic)
        #     # 或直接在 tqdm 中不指定 total，让它显示迭代进度
        #     with open(bin_path, 'wb') as bin_file:
        #         # for connection, timestamp, rawdata in tqdm(reader.messages(), desc="转换进度"):
        #         #     if connection.topic == topic:
        #         #         ros_msg = reader.deserialize(rawdata, connection.msgtype)
        #         #         events_data_list = process_func(ros_msg)
        #         #         for event_data in events_data_list:
        #         #             packed_data = struct.pack(bin_format, *event_data)
        #         #             bin_file.write(packed_data)
        # print(f"成功将事件转换并保存到 {bin_path}")


    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == '__main__':
    # 用户指定的起止时间戳（秒为单位）
    start_time = 1751629506.66296421
    end_time = 1751629506.100296421

    convert_rosbag_to_bin(ROS_BAG_PATH, EVENT_TOPIC, OUTPUT_BIN_PATH, process_event_message, BINARY_FORMAT, start_time, end_time)