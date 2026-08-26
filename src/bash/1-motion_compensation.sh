#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd -- "${script_dir}/../.." && pwd)"
if [[ -f "${workspace_root}/devel/setup.bash" ]]; then
  source "${workspace_root}/devel/setup.bash"
fi

exec roslaunch datasync Motion_Compensation_Event.launch
