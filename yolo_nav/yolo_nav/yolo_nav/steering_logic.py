"""Pure-Python YOLO-detection steering logic for the YOLO navigation branch.

No dependency on ROS 2 (rclpy), OpenCV, ``cv_bridge``, or ``ultralytics`` —
only plain Python — so it can be unit tested without a running ROS 2 /
Gazebo environment or a YOLO model checkpoint. It is imported by
``yolo_nav_node.py`` at runtime.
"""

from __future__ import annotations

from typing import Optional, TypedDict


class Detection(TypedDict):
    center_x: float
    area: float


def best_detection(detections) -> Optional[Detection]:
    """Return the detection with the largest bounding-box area, or ``None``.

    ``detections`` is an iterable of dicts with ``center_x`` and ``area``
    keys (already filtered to the target class and confidence threshold),
    mirroring the per-frame candidate list built in
    ``YoloNavigator.image_callback``.
    """
    best = None
    best_area = 0.0
    for det in detections:
        if det['area'] > best_area:
            best_area = det['area']
            best = det
    return best


def choose_steering_and_speed(
    detection: Optional[Detection],
    image_center_x: float,
    close_area: float = 15000.0,
    center_threshold: float = 40.0,
    turn_gain: float = 0.6,
    close_turn_gain: float = 0.8,
    forward_speed: float = 0.15,
    approach_speed: float = 0.10,
    search_steering: float = 0.5,
):
    """Decide (steering, speed, status) from the best chair detection.

    Direct extraction of the decision tree in
    ``YoloNavigator.image_callback`` (unchanged behavior):

    - No detection at all -> spin in place searching.
    - Detection very close (large bounding-box area) -> stop and turn away.
    - Detection left/right of the center deadzone -> turn toward it while
      creeping forward.
    - Detection centered -> drive straight forward.

    Returns:
        (steering, speed, status) tuple, where ``status`` is one of
        ``'SEARCHING'``, ``'TOO_CLOSE'``, ``'TURN_LEFT'``, ``'TURN_RIGHT'``,
        ``'FORWARD'``.
    """
    if detection is None:
        return search_steering, 0.0, 'SEARCHING'

    offset = detection['center_x'] - image_center_x
    area = detection['area']

    if area > close_area:
        return close_turn_gain, 0.0, 'TOO_CLOSE'
    if offset < -center_threshold:
        return -turn_gain, approach_speed, 'TURN_LEFT'
    if offset > center_threshold:
        return turn_gain, approach_speed, 'TURN_RIGHT'
    return 0.0, forward_speed, 'FORWARD'
