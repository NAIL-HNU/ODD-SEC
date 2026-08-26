#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <input.bag> <output.dat> [sample-rate]" >&2
  exit 1
fi

exec rosrun event_converter event_bag_to_dat "$1" "$2" "${3:-10}"
