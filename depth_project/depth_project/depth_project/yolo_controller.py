#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

# ⚠️ Change this import if needed
from yolo_msgs.msg import DetectionArray


class YoloController(Node):
    def __init__(self):
        super().__init__('yolo_controller')

        self.subscription = self.create_subscription(
            DetectionArray,
            '/yolo/detections',
            self.detection_callback,
            10
        )

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.image_width = 640
        self.last_detection = None

        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('YOLO Controller Started')

    def detection_callback(self, msg):
        if len(msg.detections) > 0:
            self.last_detection = msg.detections[0]
        else:
            self.last_detection = None

    def control_loop(self):
        cmd = Twist()

        if self.last_detection is None:
            # No object → go forward
            cmd.linear.x = 0.15
            cmd.angular.z = 0.0

        else:
            # ⚠️ Adjust depending on your message structure
            cx = self.last_detection.bbox.center.position.x

            width = self.last_detection.bbox.size.x
            height = self.last_detection.bbox.size.y
            area = width * height

            center_min = self.image_width * 0.4
            center_max = self.image_width * 0.6

            if area < 5000:
                # Far object → keep moving
                cmd.linear.x = 0.12
                cmd.angular.z = 0.0
            else:
                # Close object → turn
                cmd.linear.x = 0.0

                if cx < center_min:
                    cmd.angular.z = -0.5  # turn right
                elif cx > center_max:
                    cmd.angular.z = 0.5   # turn left
                else:
                    cmd.angular.z = 0.6   # strong turn

        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = YoloController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
