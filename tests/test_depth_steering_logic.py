"""Unit tests for depth_project.steering_logic (self-supervised depth branch).

Pure-Python / NumPy logic only, no ROS 2, OpenCV, or PyTorch required.
"""

import numpy as np
import pytest
from depth_project.steering_logic import (
    choose_steering,
    median_filter,
    percentile_of_valid,
    split_regions,
)


class TestSplitRegions:
    def test_shapes_sum_to_full_width(self):
        depth = np.ones((100, 90))
        left, center, right = split_regions(depth)
        assert left.shape[1] + center.shape[1] + right.shape[1] == 90

    def test_vertical_crop_uses_default_fractions(self):
        depth = np.arange(100 * 60).reshape(100, 60).astype(float)
        left, center, right = split_regions(depth)
        expected_rows = int(100 * 0.60) - int(100 * 0.25)
        assert left.shape[0] == expected_rows
        assert center.shape[0] == expected_rows
        assert right.shape[0] == expected_rows

    def test_custom_fractions(self):
        depth = np.ones((10, 100))
        left, center, right = split_regions(depth, left_frac=0.5, right_frac=0.5)
        # left/right split exactly in half, center band is empty
        assert left.shape[1] == 50
        assert right.shape[1] == 50
        assert center.shape[1] == 0


class TestPercentileOfValid:
    def test_ignores_nonpositive_and_nonfinite(self):
        region = np.array([1.0, 2.0, -1.0, 0.0, np.nan, np.inf, 3.0])
        # valid values are [1, 2, 3]; 50th percentile == 2
        assert percentile_of_valid(region, 50) == pytest.approx(2.0)

    def test_all_invalid_returns_zero(self):
        region = np.array([0.0, -5.0, np.nan, np.inf])
        assert percentile_of_valid(region, 50) == 0.0

    def test_matches_numpy_percentile_on_clean_data(self):
        region = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert percentile_of_valid(region, 35) == pytest.approx(np.percentile(region, 35))


class TestMedianFilter:
    def test_returns_median_of_buffer(self):
        buf = []
        assert median_filter(buf, 5.0, window=3) == 5.0
        assert median_filter(buf, 1.0, window=3) == 3.0  # median of [5, 1] == 3
        assert median_filter(buf, 3.0, window=3) == 3.0  # median of [5, 1, 3] == 3

    def test_window_caps_buffer_length(self):
        buf = []
        window = 3
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            median_filter(buf, v, window=window)
        assert len(buf) == window
        assert buf == [3.0, 4.0, 5.0]  # oldest entries popped FIFO

    def test_mutates_buffer_in_place(self):
        buf = [10.0]
        median_filter(buf, 20.0, window=5)
        assert buf == [10.0, 20.0]


class TestChooseSteering:
    def test_turns_when_obstacle_is_directly_ahead(self):
        # center well below min(left, right) * 0.85 margin: something is
        # close dead ahead, so this must NOT be 0.0 (straight into it).
        steering = choose_steering(left_depth=5.0, center_depth=1.0, right_depth=5.0)
        assert steering != 0.0

    def test_goes_straight_in_a_wide_open_room(self):
        # nothing closer ahead than to either side: no reason to turn.
        assert choose_steering(3.0, 3.0, 3.0) == 0.0

    def test_goes_straight_when_center_reads_more_open_than_the_sides(self):
        # center is farther away than both sides: clearly not blocked.
        assert choose_steering(left_depth=2.0, center_depth=5.0, right_depth=2.0) == 0.0

    def test_turns_toward_the_more_open_left_side(self):
        # center blocked, and left has more room (larger depth) than right:
        # turn toward the open side (left), not into the near one (right).
        steering = choose_steering(left_depth=5.0, center_depth=1.0, right_depth=2.0)
        assert steering == 0.8

    def test_turns_toward_the_more_open_right_side(self):
        steering = choose_steering(left_depth=2.0, center_depth=1.0, right_depth=5.0)
        assert steering == -0.8

    def test_custom_margin_can_keep_it_straight(self):
        # center (5.0) is not below min(5.0, 1.0) * 0.5 = 0.5, so a looser
        # margin reads this as open enough despite the near right side.
        steering = choose_steering(
            left_depth=5.0, center_depth=5.0, right_depth=1.0,
            center_margin=0.5, turn_gain=1.0,
        )
        assert steering == 0.0

    def test_custom_gain_applies_once_a_turn_is_warranted(self):
        steering = choose_steering(
            left_depth=5.0, center_depth=0.1, right_depth=1.0,
            center_margin=0.5, turn_gain=1.0,
        )
        assert steering == 1.0

    def test_equal_side_depths_default_to_left_on_tiebreak(self):
        steering = choose_steering(left_depth=3.0, center_depth=0.5, right_depth=3.0)
        assert steering == 0.8
