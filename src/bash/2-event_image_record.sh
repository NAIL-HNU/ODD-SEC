#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <input.bag> <output-prefix>" >&2
  exit 1
fi

input_bag="$1"
output_prefix="$2"
rosbag record /count_image -O "${output_prefix}_image.bag" &
record_pid=$!
trap 'kill "${record_pid}" 2>/dev/null || true' EXIT
sleep 1
rosbag play -r 0.1 "${input_bag}"
