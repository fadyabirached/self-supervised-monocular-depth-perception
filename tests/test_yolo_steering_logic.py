"""Unit tests for yolo_nav.steering_logic (YOLO chair-navigation branch).

Pure-Python logic only, no ROS 2, OpenCV, cv_bridge, or ultralytics required.
"""

from yolo_nav.steering_logic import best_detection, choose_steering_and_speed


class TestBestDetection:
    def test_no_detections_returns_none(self):
        assert best_detection([]) is None

    def test_picks_largest_area(self):
        dets = [
            {'center_x': 10.0, 'area': 100.0},
            {'center_x': 20.0, 'area': 500.0},
            {'center_x': 30.0, 'area': 250.0},
        ]
        assert best_detection(dets) == {'center_x': 20.0, 'area': 500.0}

    def test_single_detection(self):
        dets = [{'center_x': 5.0, 'area': 1.0}]
        assert best_detection(dets) == dets[0]


class TestChooseSteeringAndSpeed:
    def test_no_detection_searches(self):
        steer, speed, status = choose_steering_and_speed(None, image_center_x=320.0)
        assert (steer, speed, status) == (0.5, 0.0, 'SEARCHING')

    def test_close_detection_turns_and_stops(self):
        det = {'center_x': 320.0, 'area': 20000.0}
        steer, speed, status = choose_steering_and_speed(det, image_center_x=320.0)
        assert (steer, speed, status) == (0.8, 0.0, 'TOO_CLOSE')

    def test_detection_left_of_center_turns_left(self):
        # center_x well left of image center -> negative offset
        det = {'center_x': 100.0, 'area': 500.0}
        steer, speed, status = choose_steering_and_speed(det, image_center_x=320.0)
        assert (steer, speed, status) == (-0.6, 0.10, 'TURN_LEFT')

    def test_detection_right_of_center_turns_right(self):
        det = {'center_x': 540.0, 'area': 500.0}
        steer, speed, status = choose_steering_and_speed(det, image_center_x=320.0)
        assert (steer, speed, status) == (0.6, 0.10, 'TURN_RIGHT')

    def test_centered_detection_goes_forward(self):
        det = {'center_x': 325.0, 'area': 500.0}
        steer, speed, status = choose_steering_and_speed(det, image_center_x=320.0)
        assert (steer, speed, status) == (0.0, 0.15, 'FORWARD')

    def test_offset_exactly_at_threshold_is_still_centered(self):
        # offset == center_threshold (40.0) should NOT trigger a turn
        # (original code uses strict > / <)
        det = {'center_x': 360.0, 'area': 500.0}
        _steer, _speed, status = choose_steering_and_speed(det, image_center_x=320.0)
        assert status == 'FORWARD'

    def test_custom_thresholds(self):
        det = {'center_x': 350.0, 'area': 100.0}
        _steer, _speed, status = choose_steering_and_speed(
            det, image_center_x=320.0, close_area=50.0, center_threshold=10.0,
        )
        # area (100) > close_area (50) -> TOO_CLOSE takes priority
        assert status == 'TOO_CLOSE'
