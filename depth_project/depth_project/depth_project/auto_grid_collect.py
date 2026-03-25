import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Bool
from sensor_msgs.msg import LaserScan


class AutoGridCollect(Node):
    def __init__(self):
        super().__init__('auto_grid_collect')

        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.cap_pub = self.create_publisher(Bool, '/capture_image', 10)

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_cb,
            10
        )

        self.timer = self.create_timer(0.1, self.loop)

        self.front_min = 999.0
        self.left_min = 999.0
        self.right_min = 999.0
        self.scan_ready = False

        self.state = 'move'
        self.steps_left = 25

        self.capture_count = 0
        self.max_captures_per_stop = 8

        self.stop_points_done = 0
        self.max_stop_points = 15

        self.turn_dir = 1.0

        self.get_logger().info('Obstacle-aware auto grid collection started.')

    def cmd(self, lin, ang):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = float(lin)
        msg.twist.angular.z = float(ang)
        return msg

    def publish_capture(self):
        msg = Bool()
        msg.data = True
        self.cap_pub.publish(msg)

    def min_valid(self, arr):
        vals = [x for x in arr if math.isfinite(x) and x > 0.0]
        if not vals:
            return 999.0
        return min(vals)

    def scan_cb(self, msg: LaserScan):
        n = len(msg.ranges)
        if n == 0:
            return

        # Front = around 0 degrees
        front = msg.ranges[-20:] + msg.ranges[:20]

        # Left = around +90 degrees
        left_start = int(n * 0.20)
        left_end = int(n * 0.35)
        left = msg.ranges[left_start:left_end]

        # Right = around -90 degrees
        right_start = int(n * 0.65)
        right_end = int(n * 0.80)
        right = msg.ranges[right_start:right_end]

        self.front_min = self.min_valid(front)
        self.left_min = self.min_valid(left)
        self.right_min = self.min_valid(right)
        self.scan_ready = True

    def loop(self):
        if not self.scan_ready:
            self.cmd_pub.publish(self.cmd(0.0, 0.0))
            return

        if self.stop_points_done >= self.max_stop_points:
            self.cmd_pub.publish(self.cmd(0.0, 0.0))
            return

        # If obstacle is too close in front, escape turn
        if self.front_min < 0.45 and self.state in ['move', 'settle']:
            self.state = 'escape_turn'
            self.steps_left = 12
            self.turn_dir = -1.0 if self.right_min < self.left_min else 1.0

        if self.state == 'move':
            # Move only if front is reasonably free
            if self.front_min > 0.60:
                self.cmd_pub.publish(self.cmd(0.10, 0.0))
                self.steps_left -= 1
            else:
                self.state = 'escape_turn'
                self.steps_left = 12
                self.turn_dir = -1.0 if self.right_min < self.left_min else 1.0

            if self.steps_left <= 0:
                self.state = 'settle'
                self.steps_left = 8

        elif self.state == 'settle':
            self.cmd_pub.publish(self.cmd(0.0, 0.0))
            self.steps_left -= 1
            if self.steps_left <= 0:
                self.state = 'capture'
                self.steps_left = 5

        elif self.state == 'capture':
            self.cmd_pub.publish(self.cmd(0.0, 0.0))
            self.steps_left -= 1
            if self.steps_left <= 0:
                self.publish_capture()
                self.capture_count += 1

                if self.capture_count >= self.max_captures_per_stop:
                    self.capture_count = 0
                    self.stop_points_done += 1
                    self.state = 'move'
                    self.steps_left = 25
                else:
                    self.state = 'rotate'
                    self.steps_left = 4

        elif self.state == 'rotate':
            self.cmd_pub.publish(self.cmd(0.0, 0.35))
            self.steps_left -= 1
            if self.steps_left <= 0:
                self.state = 'settle'
                self.steps_left = 6

        elif self.state == 'escape_turn':
            self.cmd_pub.publish(self.cmd(0.0, self.turn_dir * 0.55))
            self.steps_left -= 1
            if self.steps_left <= 0:
                self.state = 'move'
                self.steps_left = 18


def main(args=None):
    rclpy.init(args=args)
    node = AutoGridCollect()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.cmd_pub.publish(node.cmd(0.0, 0.0))
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
