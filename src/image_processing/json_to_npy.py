import os
import json
import numpy as np
import argparse

# 定义结构化数组的数据类型
dtype = np.dtype([
    ('ts', '<u8'),       # 时间戳，64 位无符号整数
    ('x', '<f4'),       # 边界框左上角 x 坐标，32 位浮点数
    ('y', '<f4'),       # 边界框左上角 y 坐标，32 位浮点数
    ('w', '<f4'),       # 边界框宽度，32 位浮点数
    ('h', '<f4'),       # 边界框高度，32 位浮点数
    ('class_id', 'u1'), # 类别 ID，8 位无符号整数
    ('confidence', '<f4'), # 置信度，32 位浮点数
    ('track_id', '<u4') # 跟踪 ID，32 位无符号整数
])

# 定义 class_id 和 confidence 的固定值
CLASS_ID = 0
CONFIDENCE = 1.0

def json_to_npy(json_folder, output_npy_path):
    """
    将 JSON 文件夹中的所有 JSON 文件转换为一个 .npy 文件。
    
    Args:
        json_folder (str): JSON 文件所在的文件夹路径。
        output_npy_path (str): 输出的 .npy 文件路径。
    """
    # 初始化一个空列表，用于存储所有数据
    data_list = []

    # 初始化 track_id 计数器
    track_id_counter = 1

    # 获取文件夹中的所有文件，并按文件名排序
    sorted_files = sorted(os.listdir(json_folder), key=lambda x: int(os.path.splitext(x)[0]))

    # 按顺序遍历文件
    for filename in sorted_files:
        if filename.endswith(".json"):
            json_path = os.path.join(json_folder, filename)

            # 读取 JSON 文件
            with open(json_path, "r") as f:
                json_data = json.load(f)

            # 提取 shapes 中的标注框信息
            for shape in json_data["shapes"]:
                # 提取边界框的左上角和右下角坐标
                (x1, y1), (x2, y2) = shape["points"]
                # 计算边界框的宽度 (w) 和高度 (h)
                w = abs(x2 - x1)
                h = abs(y2 - y1)

                # 从文件名中提取 ts（假设文件名是 xxxx.json）
                ts = int(os.path.splitext(filename)[0])  # 去掉扩展名并转换为整数

                # 使用递增的 track_id
                track_id = track_id_counter
                track_id_counter += 1

                # 将数据添加到列表中
                data_list.append((ts, x1, y1, w, h, CLASS_ID, CONFIDENCE, track_id))

    # 将列表转换为 NumPy 结构化数组
    data_array = np.array(data_list, dtype=dtype)

    # 保存为 .npy 文件
    np.save(output_npy_path, data_array)
    print(f"Saved {len(data_list)} boxes to {output_npy_path}")


if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='Convert JSON files to a single .npy file.')
    parser.add_argument('json_folder', type=str, help='Path to the folder containing JSON files')
    parser.add_argument('output_npy_path', type=str, help='Path to the output .npy file')
    args = parser.parse_args()

    # 调用函数
    json_to_npy(args.json_folder, args.output_npy_path)