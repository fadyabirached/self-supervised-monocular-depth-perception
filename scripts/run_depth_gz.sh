#!/usr/bin/env bash
#
# Run the self-supervised depth branch on modern gz sim (Gazebo Harmonic).
#
# This is the counterpart to run_depth.sh, which targets classic Gazebo.
# Classic reached end of life in January 2025 and has no package on Ubuntu
# 24.04 / ROS 2 Jazzy, so on a current install this is the script that
# actually runs. It launches depth_project/launch/columns_world.launch.py:
# the same arena (columns_world.sdf), the TurtleBot3 waffle with its
# camera, the two ros_gz bridges, then depth_node and robot_controller.
#
# Prerequisites:
#   - ROS 2 Jazzy + ros_gz (see README "Environment"), or the container in
#     .devcontainer/, which has all of it preinstalled
#   - Workspace built, e.g. via scripts/setup_workspace.sh
#   - CPU torch:  pip install --index-url https://download.pytorch.org/whl/cpu torch
#
# Env vars:
#   ROS_DISTRO      ROS 2 distro to source (default: jazzy)
#   ROS2_WS         Colcon workspace to source (default: ~/ros2_ws)
#   HEADLESS        true to run without the Gazebo GUI (default: false)
#   RENDER_ENGINE   ogre2 (default) or ogre. On a machine with no GPU,
#                   ogre software-rasterizes far more reliably than ogre2,
#                   so try RENDER_ENGINE=ogre if Gazebo fails to start.

set -euo pipefail

ROS_DISTRO="${ROS_DISTRO:-jazzy}"
ROS2_WS="${ROS2_WS:-$HOME/ros2_ws}"
HEADLESS="${HEADLESS:-false}"
RENDER_ENGINE="${RENDER_ENGINE:-ogre2}"

# shellcheck disable=SC1090
source "/opt/ros/${ROS_DISTRO}/setup.bash"
# shellcheck disable=SC1090
source "${ROS2_WS}/install/setup.bash"

exec ros2 launch depth_project columns_world.launch.py \
    headless:="${HEADLESS}" \
    render_engine:="${RENDER_ENGINE}"
