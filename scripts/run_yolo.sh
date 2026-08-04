#!/usr/bin/env bash
#
# Run the YOLO object-detection obstacle-avoidance branch:
#   TurtleBot3 (waffle) in the new "gz sim" simulator (Gazebo Harmonic),
#   custom "yolo_world" (four walls + four office chairs), driven by
#   yolo_nav_node (YOLOv8n chair detection + steering) and
#   robot_controller (turns /steering_cmd into /model/waffle/cmd_vel).
#
# Unlike the depth branch, this is composed of two separate launches:
#   1. my_yolo_world/launch/tb3_custom_world.launch.py
#        gz sim + robot_state_publisher + robot spawn + cmd_vel/image bridges
#   2. depth_project's robot_controller + yolo_nav's yolo_nav_node
#        (run as plain nodes, started once the world above is up)
#
# Prerequisites (see README "Environment"):
#   - ROS 2 Jazzy + Gazebo Harmonic (gz sim) + ros_gz_sim/ros_gz_bridge/
#     ros_gz_image + turtlebot3_gazebo/turtlebot3_description installed
#   - Workspace built, e.g. via scripts/setup_workspace.sh
#   - pip install -r requirements.txt (includes ultralytics for YOLOv8n,
#     which is downloaded automatically on first run)
#
# Known limitation: my_yolo_world/launch/tb3_custom_world.launch.py
# hardcodes the world file path to ~/ros2_ws/src/my_yolo_world/worlds/
# yolo_world.sdf, so this script (and setup_workspace.sh) assume the
# default ~/ros2_ws workspace layout.
#
# Env vars:
#   ROS_DISTRO   ROS 2 distro to source (default: jazzy)
#   ROS2_WS      Colcon workspace to source (default: ~/ros2_ws)

set -euo pipefail

ROS_DISTRO="${ROS_DISTRO:-jazzy}"
ROS2_WS="${ROS2_WS:-$HOME/ros2_ws}"

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/ros_env.sh"
require_workspace "${ROS2_WS}"
source_ros "${ROS_DISTRO}" "${ROS2_WS}"

pids=()
cleanup() {
    echo "Shutting down..."
    for pid in "${pids[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

echo "Starting Gazebo (yolo_world) + bridges..."
ros2 launch my_yolo_world tb3_custom_world.launch.py &
pids+=($!)

# tb3_custom_world.launch.py stages up to ~12s of TimerActions itself
# (world -> robot_state_publisher -> spawn -> bridges); give it margin
# before starting the perception/control nodes.
sleep 15

echo "Starting robot_controller (steering_cmd -> cmd_vel)..."
ros2 run depth_project robot_controller &
pids+=($!)

echo "Starting yolo_nav_node (YOLOv8n chair detection -> steering_cmd)..."
ros2 run yolo_nav yolo_nav_node &
pids+=($!)

wait
