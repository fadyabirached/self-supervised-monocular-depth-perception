"""Test path setup.

This repo is a ROS 2 workspace with three colcon packages
(``depth_project``, ``my_yolo_world``, ``yolo_nav``), each nested one
level deeper than a normal Python project
(``<pkg>/<pkg>/<python_package>/``). These tests only exercise the pure
Python decision-logic modules (no ROS 2 / OpenCV / PyTorch / Gazebo
required), so we add the relevant package source roots to ``sys.path``
directly instead of requiring a full ``colcon build``.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

for pkg_root in (
    REPO_ROOT / 'depth_project' / 'depth_project',
    REPO_ROOT / 'yolo_nav' / 'yolo_nav',
):
    path_str = str(pkg_root)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
