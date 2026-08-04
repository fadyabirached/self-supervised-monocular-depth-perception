"""Pure-Python depth-map steering logic for the self-supervised branch.

This module intentionally has **no** dependency on ROS 2 (rclpy), OpenCV,
or PyTorch. It only needs NumPy. It is imported by ``depth_node.py`` at
runtime, but is kept separate so the decision logic can be unit tested
without a running ROS 2 / Gazebo / GPU environment.
"""

from __future__ import annotations

import numpy as np


def split_regions(depth_map, y1_frac: float = 0.25, y2_frac: float = 0.60,
                   left_frac: float = 0.33, right_frac: float = 0.67):
    """Split a 2D depth map into left/center/right regions of interest.

    A horizontal strip of the image (between ``y1_frac`` and ``y2_frac`` of
    the height) is taken, then split into three vertical bands.

    Args:
        depth_map: 2D array-like of shape (H, W).
        y1_frac: top of the strip, as a fraction of image height.
        y2_frac: bottom of the strip, as a fraction of image height.
        left_frac: right edge of the "left" band, as a fraction of width.
        right_frac: left edge of the "right" band, as a fraction of width.

    Returns:
        (left, center, right) sub-arrays.
    """
    depth_map = np.asarray(depth_map)
    h, _w = depth_map.shape
    y1 = int(h * y1_frac)
    y2 = int(h * y2_frac)
    roi = depth_map[y1:y2, :]
    width = roi.shape[1]

    left = roi[:, :int(width * left_frac)]
    center = roi[:, int(width * left_frac):int(width * right_frac)]
    right = roi[:, int(width * right_frac):]
    return left, center, right


def percentile_of_valid(region, q: float) -> float:
    """Return the ``q``-th percentile of the finite, positive values in ``region``.

    Non-finite values (NaN/inf) and non-positive depths are treated as
    invalid sensor readings and excluded. Returns 0.0 if nothing is valid.
    """
    arr = np.asarray(region).flatten()
    valid = arr[np.isfinite(arr) & (arr > 0)]
    return float(np.percentile(valid, q)) if len(valid) > 0 else 0.0


def median_filter(buffer: list, value: float, window: int) -> float:
    """Push ``value`` onto ``buffer`` (capped at ``window`` entries, FIFO) and
    return the median of the buffer.

    ``buffer`` is mutated in place, mirroring the temporal smoothing used in
    ``DepthNode`` to reduce single-frame noise in the steering signal.
    """
    buffer.append(value)
    if len(buffer) > window:
        buffer.pop(0)
    return float(np.median(buffer))


def choose_steering(left_depth: float, center_depth: float, right_depth: float,
                     center_margin: float = 0.85, turn_gain: float = 0.8) -> float:
    """Choose a steering command from three smoothed region depths.

    Depth values are metres to the nearest obstacle in that region
    (see ``losses.disp_to_depth``): smaller means nearer/more blocked,
    larger means farther/more open. Go straight while the center reads
    at least as open as the more cautious of the two sides; once it
    reads clearly nearer than both (scaled by ``center_margin``), turn
    toward whichever side has more room.

    This was previously inverted on both axes: it drove straight at
    whatever was directly ahead, and turned toward whichever side had
    the *nearer* obstacle rather than away from it. Two things gave it
    away. First, the module and README both describe the opposite of
    what the old comparisons did ("go straight unless the center is
    clearly closer... otherwise turn toward whichever side reads more
    open" is not what `if center < ... : return 0.0` implements).
    Second, driving the real simulation reproduced it exactly: the
    robot rammed columns it was facing head-on, and idled in the open
    turning left forever because an all-clear reading also used to
    satisfy the (inverted) turn condition.

    Returns a steering value of either ``0.0``, ``turn_gain`` or
    ``-turn_gain``.
    """
    if center_depth >= min(left_depth, right_depth) * center_margin:
        return 0.0
    return turn_gain if left_depth >= right_depth else -turn_gain
