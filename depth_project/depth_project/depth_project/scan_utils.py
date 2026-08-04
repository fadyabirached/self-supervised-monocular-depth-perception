"""Pure-Python LaserScan helpers used by auto_grid_collect.py.

No ROS 2 dependency, only needs the standard library ``math`` module, so
it can be unit tested without a running ROS 2 / Gazebo environment.
"""

from __future__ import annotations

import math


def min_valid_range(ranges, default: float = 999.0) -> float:
    """Return the minimum finite, positive value in a LaserScan ``ranges`` list.

    ``inf``/``nan`` entries (out-of-range or invalid readings) and
    non-positive values are ignored. If nothing is valid, ``default`` is
    returned (matching the "nothing detected nearby" sentinel used in
    ``AutoGridCollect``).
    """
    valid = [x for x in ranges if math.isfinite(x) and x > 0.0]
    return min(valid) if valid else default
