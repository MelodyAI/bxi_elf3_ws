#!/usr/bin/env bash
set -e
# Usage: ./scripts/run_debug_bxi_example.sh
# This script sources the local workspace install and launches the node
# under debugpy so VSCode can attach (port 5678).

WS_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ -f "$WS_ROOT/install/setup.bash" ]; then
  # Source workspace overlay (preferred)
  # shellcheck disable=SC1091
  source "$WS_ROOT/install/setup.bash"
  echo "Sourced $WS_ROOT/install/setup.bash"
else
  echo "Warning: $WS_ROOT/install/setup.bash not found. Make sure ROS2 and workspace are sourced."
fi

# Ensure python packages from install are found
export PYTHONPATH="$WS_ROOT/install/lib/python3.10/site-packages:$PYTHONPATH"

SCRIPT="$WS_ROOT/src/bxi_example_py_elf3/bxi_example_py_elf3/bxi_example_dance.py"

if [ ! -f "$SCRIPT" ]; then
  echo "Cannot find node script: $SCRIPT"
  exit 1
fi

echo "Starting node under debugpy, waiting for VSCode to attach on port 5678..."
exec python3 -m debugpy --listen 5678 --wait-for-client "$SCRIPT" "$@"
