#!/usr/bin/env bash
#
# One-time setup: link this repo's three ROS 2 packages into a colcon
# workspace and build them.
#
# Some launch files in this repo (see README "Known limitations") use
# absolute paths under ~/ros2_ws/src, e.g. the world file path in
# my_yolo_world/launch/tb3_custom_world.launch.py, and the checkpoint
# path in depth_project/depth_project/depth_node.py. Using the default
# ~/ros2_ws workspace location (or overriding via $ROS2_WS below and
# updating those paths yourself) avoids surprises.
#
# Usage:
#   scripts/setup_workspace.sh
#
# Env vars:
#   ROS_DISTRO   ROS 2 distro to source (default: jazzy)
#   ROS2_WS      Target colcon workspace (default: ~/ros2_ws)

set -euo pipefail

ROS_DISTRO="${ROS_DISTRO:-jazzy}"
ROS2_WS="${ROS2_WS:-$HOME/ros2_ws}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Sourcing ROS 2 ${ROS_DISTRO}..."
# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/ros_env.sh"
source_ros "${ROS_DISTRO}" "${ROS2_WS}"

mkdir -p "${ROS2_WS}/src"

link_package () {
    local name="$1"
    local src="${REPO_ROOT}/${name}/${name}"
    local dst="${ROS2_WS}/src/${name}"

    if [ -e "${dst}" ] && [ ! -L "${dst}" ]; then
        echo "Refusing to overwrite existing non-symlink at ${dst}" >&2
        exit 1
    fi

    ln -sfn "${src}" "${dst}"
    echo "Linked ${dst} -> ${src}"
}

link_package depth_project
link_package my_yolo_world
link_package yolo_nav

echo "Building workspace at ${ROS2_WS}..."
cd "${ROS2_WS}"
colcon build --symlink-install --packages-select depth_project my_yolo_world yolo_nav

cat <<EOF

Done. Before running the demos:
  1. Install the pip dependencies:  pip install -r ${REPO_ROOT}/requirements.txt
  2. Train (or otherwise obtain) a depth checkpoint at:
       ${ROS2_WS}/src/depth_project/checkpoints/selfsup_depth_latest.pth
     (a placeholder checkpoint already ships in this repo at
     depth_project/depth_project/checkpoints/selfsup_depth_latest.pth,
     which the symlink above already exposes at that path)
  3. Run scripts/run_depth.sh or scripts/run_yolo.sh
EOF
