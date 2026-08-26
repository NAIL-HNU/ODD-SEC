import os
import json
import numpy as np
import argparse

dtype = np.dtype([
    ('ts', '<u8'),
    ('x', '<f4'),
    ('y', '<f4'),
    ('w', '<f4'),
    ('h', '<f4'),
    ('class_id', 'u1'),
    ('confidence', '<f4'),
    ('track_id', '<u4')
])

CLASS_ID = 0
CONFIDENCE = 1.0

def json_to_npy(json_folder, output_npy_path):
    """Convert a directory of JSON annotations to one NumPy array."""
    data_list = []

    track_id_counter = 1

    sorted_files = sorted(os.listdir(json_folder), key=lambda x: int(os.path.splitext(x)[0]))

    for filename in sorted_files:
        if filename.endswith(".json"):
            json_path = os.path.join(json_folder, filename)

            with open(json_path, "r") as f:
                json_data = json.load(f)

            for shape in json_data["shapes"]:
                (x1, y1), (x2, y2) = shape["points"]
                w = abs(x2 - x1)
                h = abs(y2 - y1)

                ts = int(os.path.splitext(filename)[0])

                track_id = track_id_counter
                track_id_counter += 1

                data_list.append((ts, x1, y1, w, h, CLASS_ID, CONFIDENCE, track_id))

    data_array = np.array(data_list, dtype=dtype)

    np.save(output_npy_path, data_array)
    print(f"Saved {len(data_list)} boxes to {output_npy_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert JSON files to a single .npy file.')
    parser.add_argument('json_folder', type=str, help='Path to the folder containing JSON files')
    parser.add_argument('output_npy_path', type=str, help='Path to the output .npy file')
    args = parser.parse_args()

    json_to_npy(args.json_folder, args.output_npy_path)
