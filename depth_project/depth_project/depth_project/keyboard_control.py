import sys
import termios
import tty
import select
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped


class KeyboardControl(Node):
    def __init__(self):
        super().__init__('keyboard_control')

        self.pub = self.create_publisher(
            TwistStamped,
            '/model/waffle/cmd_vel',
            10
        )

        self.speed = 0.0
        self.steering = 0.0
        self.timer = self.create_timer(0.1, self.publish_cmd)

        self.get_logger().info(
            "\nKeyboard control:\n"
            "  w : move forward\n"
            "  s : stop\n"
            "  a : steer left\n"
            "  d : steer right\n"
            "  e : straighten\n"
            "  q : quit\n"
        )

    def publish_cmd(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = self.speed
        msg.twist.angular.z = 1.8 * self.steering
        self.pub.publish(msg)

    def get_key(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
            return sys.stdin.read(1) if rlist else ''
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def run(self):
        while rclpy.ok():
            key = self.get_key()
            if key == 'w':
                self.speed = 0.25
            elif key == 's':
                self.speed = 0.0
            elif key == 'a':
                self.steering = 0.8
            elif key == 'd':
                self.steering = -0.8
            elif key == 'e':
                self.steering = 0.0
            elif key == 'q':
                break
            rclpy.spin_once(self, timeout_sec=0.01)


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardControl()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
