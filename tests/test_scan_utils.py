"""Unit tests for depth_project.scan_utils (auto_grid_collect LaserScan helper).

Pure-Python logic only — no ROS 2 required.
"""

import math

from depth_project.scan_utils import min_valid_range


def test_returns_minimum_of_valid_values():
    assert min_valid_range([3.0, 1.5, 2.0]) == 1.5


def test_ignores_nonpositive_values():
    assert min_valid_range([-1.0, 0.0, 4.0]) == 4.0


def test_ignores_inf_and_nan():
    assert min_valid_range([math.inf, math.nan, 2.5, math.inf]) == 2.5


def test_returns_default_when_nothing_valid():
    assert min_valid_range([math.inf, -1.0, 0.0]) == 999.0


def test_custom_default():
    assert min_valid_range([], default=42.0) == 42.0


def test_single_valid_value():
    assert min_valid_range([math.inf, 7.0, math.inf]) == 7.0
