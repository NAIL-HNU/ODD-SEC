import numpy as np
import os
import glob
import cv2
import argparse

def npy_to_bin(npy_file_path, output_bin_path, target_height=480, target_width=640):
    """Convert NumPy frames to contiguous float32 TensorRT input."""
    print(f"Reading {npy_file_path}...")
    frames = np.load(npy_file_path)
    
    print(f"Input shape: {frames.shape}")
    print(f"Frames: {frames.shape[0]}, Height: {frames.shape[1]}, Width: {frames.shape[2]}")
    
    if frames.shape[0] > 0:
        first_frame = frames[0]
        
        if first_frame.dtype != np.uint8:
            if first_frame.max() > 1.0:
                first_frame_normalized = first_frame.astype(np.float32)
            else:
                first_frame_normalized = (first_frame * 255).astype(np.float32)
            first_frame_normalized = np.clip(first_frame_normalized, 0, 255).astype(np.uint8)
        else:
            first_frame_normalized = first_frame
        
        base_name = os.path.splitext(output_bin_path)[0]
        image_output_path = f"{base_name}_first_frame_original.png"
        
        cv2.imwrite(image_output_path, first_frame_normalized)
        print(f"Saved original first frame as: {image_output_path}")
        print(f"Original first frame - min: {first_frame.min():.3f}, max: {first_frame.max():.3f}, dtype: {first_frame.dtype}")
    
    frames = frames.astype(np.float32)
    
    print(f"Rotating frames from {frames.shape[2]}x{frames.shape[1]} to {target_width}x{target_height}")
    rotated_frames = []
    for i in range(frames.shape[0]):
        rotated_frames.append(frames[i])
    
    frames = np.array(rotated_frames)
    print(f"After rotation shape: {frames.shape}")
    
    
    if frames.shape[0] > 0:
        rotated_first_frame = frames[0]
        
        if rotated_first_frame.dtype != np.uint8:
            if rotated_first_frame.max() > 1.0:
                rotated_first_frame_normalized = rotated_first_frame.astype(np.float32)
            else:
                rotated_first_frame_normalized = (rotated_first_frame * 255).astype(np.float32)
            rotated_first_frame_normalized = np.clip(rotated_first_frame_normalized, 0, 255).astype(np.uint8)
        else:
            rotated_first_frame_normalized = rotated_first_frame
        
        rotated_image_output_path = f"{base_name}_first_frame_rotated.png"
        
        cv2.imwrite(rotated_image_output_path, rotated_first_frame_normalized)
        print(f"Saved rotated first frame as: {rotated_image_output_path}")
        print(f"Rotated first frame - min: {rotated_first_frame.min():.3f}, max: {rotated_first_frame.max():.3f}")

    data_flat = frames.flatten()
    
    print(f"Saving to {output_bin_path}...")
    data_flat.tofile(output_bin_path)
    
    print(f"Successfully converted! Output size: {data_flat.nbytes} bytes")
    print(f"Expected size: {10 * target_height * target_width * 4} bytes")
    
    return True

def batch_convert_npy_to_bin(input_folder, output_folder, target_height=480, target_width=640):
    """Convert every .npy file in a directory."""
    os.makedirs(output_folder, exist_ok=True)
    
    npy_files = glob.glob(os.path.join(input_folder, "*.npy"))
    
    if not npy_files:
        print(f"No .npy files found in {input_folder}")
        return
    
    print(f"Found {len(npy_files)} .npy files")
    
    success_count = 0
    for npy_file in npy_files:
        try:
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
    parser = argparse.ArgumentParser(description="Convert NumPy frames to TensorRT binary input.")
    parser.add_argument("input_path", help="Input .npy file or directory")
    parser.add_argument("output_path", help="Output .bin file or directory")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()

    if os.path.isdir(args.input_path):
        batch_convert_npy_to_bin(args.input_path, args.output_path, args.height, args.width)
    else:
        npy_to_bin(args.input_path, args.output_path, args.height, args.width)
