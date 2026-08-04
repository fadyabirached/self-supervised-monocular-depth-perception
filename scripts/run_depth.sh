#!/usr/bin/env bash
#
# Run the self-supervised monocular depth obstacle-avoidance branch:
#   TurtleBot3 (burger) in classic Gazebo, "columns_world", driven by
#   depth_node (self-supervised depth inference + region-based steering)
#   and robot_controller (turns /steering_cmd into /model/waffle/cmd_vel).
#
# This is a single ROS 2 launch file (depth_project/launch/gazebo.launch.py)
# that starts Gazebo, then depth_node after 5s, then robot_controller
# after 6s, giving the simulation time to settle.
#
# Prerequisites (see README "Environment"):
#   - ROS 2 Jazzy + classic Gazebo + turtlebot3_gazebo installed
#   - Workspace built, e.g. via scripts/setup_workspace.sh
#   - pip install -r requirements.txt
#   - A depth checkpoint at ~/ros2_ws/src/depth_project/checkpoints/selfsup_depth_latest.pth
#     (see depth_project/depth_project/checkpoints/selfsup_depth_latest.pth)
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

exec ros2 launch depth_project gazebo.launch.py
