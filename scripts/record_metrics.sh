#!/usr/bin/env bash
#
# Log obstacle-encounter outcomes for one branch, to build the
# depth-vs-YOLO comparison in the README's Observations section.
#
# This does not measure anything on its own: it is a keyboard-driven
# stopwatch. Run it in its own terminal, alongside the branch's own
# run_*.sh script (already running in another terminal) and the Gazebo
# window (open in the browser), and mark what you see as it happens:
#
#   o   an obstacle just came into the robot's path
#   c   it collided with that obstacle
#   s   it got past that obstacle without colliding
#   n   skip to the next encounter without marking success or collision
#   q   stop and save
#
# One encounter = one row. Mark 'o' the moment an obstacle is clearly
# ahead of the robot, then 's' or 'c' once the outcome is obvious. The
# node timestamps 'o' and the first steering command that moves off zero
# afterwards, so reaction_time in the CSV is measured, not guessed.
#
# The simulation keeps running between encounters, no need to restart it:
# in this arena the robot just keeps approaching the next column or wall.
#
# Usage:
#   scripts/record_metrics.sh depth
#   scripts/record_metrics.sh yolo
#   scripts/record_metrics.sh depth ~/depth_metrics.csv
#
# Env vars:
#   ROS_DISTRO   ROS 2 distro to source (default: jazzy)
#   ROS2_WS      Colcon workspace to source (default: ~/ros2_ws)

set -euo pipefail

METHOD="${1:?Usage: record_metrics.sh <depth|yolo> [csv_path]}"
CSV_PATH="${2:-$HOME/${METHOD}_metrics.csv}"

case "${METHOD}" in
    depth|yolo) ;;
    *)
        echo "method must be 'depth' or 'yolo', got '${METHOD}'" >&2
        exit 1
        ;;
esac

ROS_DISTRO="${ROS_DISTRO:-jazzy}"
ROS2_WS="${ROS2_WS:-$HOME/ros2_ws}"

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/ros_env.sh"
require_workspace "${ROS2_WS}"
source_ros "${ROS_DISTRO}" "${ROS2_WS}"

echo "Logging '${METHOD}' encounters to ${CSV_PATH}"
echo "Keys: o=obstacle appeared, c=collision, s=success, n=next, q=quit"
echo

exec ros2 run depth_project metrics_logger --ros-args \
    -p method:="${METHOD}" \
    -p csv_path:="${CSV_PATH}" \
    -p steering_threshold:=0.01
