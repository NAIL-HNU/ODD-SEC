#!/usr/bin/env python3

import argparse

import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Inspect a NumPy array file.")
    parser.add_argument("npy_path", help="Input .npy file")
    args = parser.parse_args()

    data = np.load(args.npy_path, allow_pickle=True)
    print("Data shape:", data.shape)
    print("Data type:", data.dtype)
    print("Data content:")
    print(data)

    if data.dtype.fields is not None:
        print("\nField names:", data.dtype.names)
        for field in data.dtype.names:
            print(f"Field '{field}':", data[field])

if __name__ == "__main__":
    main()
