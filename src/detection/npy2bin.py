import numpy as np
import os
import glob
import cv2  # 添加cv2导入

def npy_to_bin(npy_file_path, output_bin_path, target_height=480, target_width=640):
    """
    将.npy文件转换为.bin文件，并将横向图片旋转为竖向
    
    Args:
        npy_file_path: 输入的.npy文件路径
        output_bin_path: 输出的.bin文件路径
        target_height: 目标高度 (默认480)
        target_width: 目标宽度 (默认640)
    """
    # 读取.npy文件
    print(f"Reading {npy_file_path}...")
    frames = np.load(npy_file_path)
    
    print(f"Input shape: {frames.shape}")
    print(f"Frames: {frames.shape[0]}, Height: {frames.shape[1]}, Width: {frames.shape[2]}")
    
    # 保存第一帧为灰度图（旋转前）
    if frames.shape[0] > 0:
        first_frame = frames[0]
        
        # 归一化到0-255范围
        if first_frame.dtype != np.uint8:
            if first_frame.max() > 1.0:
                first_frame_normalized = first_frame.astype(np.float32)
            else:
                first_frame_normalized = (first_frame * 255).astype(np.float32)
            first_frame_normalized = np.clip(first_frame_normalized, 0, 255).astype(np.uint8)
        else:
            first_frame_normalized = first_frame
        
        # 生成输出图像文件名（旋转前）
        base_name = os.path.splitext(output_bin_path)[0]
        image_output_path = f"{base_name}_first_frame_original.png"
        
        # 保存为PNG图像
        cv2.imwrite(image_output_path, first_frame_normalized)
        print(f"Saved original first frame as: {image_output_path}")
        print(f"Original first frame - min: {first_frame.min():.3f}, max: {first_frame.max():.3f}, dtype: {first_frame.dtype}")
    
    # 确保数据类型为float32
    frames = frames.astype(np.float32)
    
    # 旋转每一帧：从横向(640x480)旋转为竖向(480x640)
    print(f"Rotating frames from {frames.shape[2]}x{frames.shape[1]} to {target_width}x{target_height}")
    rotated_frames = []
    for i in range(frames.shape[0]):
        # # 旋转90度逆时针，将640x480变为480x640
        # rotated = cv2.rotate(frames[i], cv2.ROTATE_90_COUNTERCLOCKWISE)
        rotated_frames.append(frames[i])
    
    frames = np.array(rotated_frames)
    print(f"After rotation shape: {frames.shape}")
    
    
    # 如果需要调整尺寸（现在应该是480x640，与目标尺寸匹配）
    # if frames.shape[1] != target_height or frames.shape[2] != target_width:
    #     print(f"Resizing from {frames.shape[1]}x{frames.shape[2]} to {target_height}x{target_width}")
    #     resized_frames = []
    #     for i in range(frames.shape[0]):
    #         # 调整尺寸
    #         resized = cv2.resize(frames[i], (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    #         resized_frames.append(resized)
    #     frames = np.array(resized_frames)
    
    # 保存旋转后的第一帧为灰度图
    if frames.shape[0] > 0:
        rotated_first_frame = frames[0]
        
        # 归一化到0-255范围
        if rotated_first_frame.dtype != np.uint8:
            if rotated_first_frame.max() > 1.0:
                rotated_first_frame_normalized = rotated_first_frame.astype(np.float32)
            else:
                rotated_first_frame_normalized = (rotated_first_frame * 255).astype(np.float32)
            rotated_first_frame_normalized = np.clip(rotated_first_frame_normalized, 0, 255).astype(np.uint8)
        else:
            rotated_first_frame_normalized = rotated_first_frame
        
        # 生成输出图像文件名（旋转后）
        rotated_image_output_path = f"{base_name}_first_frame_rotated.png"
        
        # 保存为PNG图像
        cv2.imwrite(rotated_image_output_path, rotated_first_frame_normalized)
        print(f"Saved rotated first frame as: {rotated_image_output_path}")
        print(f"Rotated first frame - min: {rotated_first_frame.min():.3f}, max: {rotated_first_frame.max():.3f}")

    # 将数据重新排列为连续的float数组
    # 从 [10, H, W] 转换为 [10*H*W] 的连续数组
    data_flat = frames.flatten()
    
    # 保存为.bin文件
    print(f"Saving to {output_bin_path}...")
    data_flat.tofile(output_bin_path)
    
    print(f"Successfully converted! Output size: {data_flat.nbytes} bytes")
    print(f"Expected size: {10 * target_height * target_width * 4} bytes")
    
    return True

def batch_convert_npy_to_bin(input_folder, output_folder, target_height=480, target_width=640):
    """
    批量转换文件夹中的所有.npy文件
    
    Args:
        input_folder: 包含.npy文件的文件夹
        output_folder: 输出.bin文件的文件夹
        target_height: 目标高度
        target_width: 目标宽度
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 查找所有.npy文件
    npy_files = glob.glob(os.path.join(input_folder, "*.npy"))
    
    if not npy_files:
        print(f"No .npy files found in {input_folder}")
        return
    
    print(f"Found {len(npy_files)} .npy files")
    
    success_count = 0
    for npy_file in npy_files:
        try:
            # 生成输出文件名
            basename = os.path.splitext(os.path.basename(npy_file))[0]
            output_bin = os.path.join(output_folder, f"{basename}.bin")
            
            print(f"\n{'='*50}")
            print(f"Converting: {basename}")
            
            if npy_to_bin(npy_file, output_bin, target_height, target_width):
                success_count += 1
                
        except Exception as e:
            print(f"Error converting {npy_file}: {e}")
    
    print(f"\n{'='*50}")
    print(f"Conversion completed: {success_count}/{len(npy_files)} files converted successfully")

if __name__ == "__main__":
    # 示例用法
    
    # 单个文件转换
    npy_to_bin("/home/zhx/dataset/Added/result/2025-08-03-18-33-23_01469017.npy", "/home/zhx/dataset/Added/result/2025-08-03-18-33-23_01469017.bin")
    
    # 批量转换
    # input_folder = "/home/zhx/result/"  # 替换为你的.npy文件路径
    # output_folder = "/home/zhx/result/"  # 替换为输出.bin文件的路径
    
    # batch_convert_npy_to_bin(input_folder, output_folder)