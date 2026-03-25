import random
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped


class AutoCollectNode(Node):
    def __init__(self):
        super().__init__('auto_collect')

        self.pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)

        self.mode = 'forward'
        self.mode_time = 0
        self.timer = self.create_timer(0.1, self.loop)

        self.get_logger().info('Auto data collection movement started.')

    def make_cmd(self, lin: float, ang: float) -> TwistStamped:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = float(lin)
        msg.twist.angular.z = float(ang)
        return msg

    def choose_new_mode(self):
        r = random.random()

        if r < 0.50:
            self.mode = 'forward'
            self.mode_time = random.randint(15, 35)
        elif r < 0.75:
            self.mode = 'left'
            self.mode_time = random.randint(10, 20)
        else:
            self.mode = 'right'
            self.mode_time = random.randint(10, 20)

    def loop(self):
        if self.mode_time <= 0:
            self.choose_new_mode()

        if self.mode == 'forward':
            cmd = self.make_cmd(0.10, 0.0)
        elif self.mode == 'left':
            cmd = self.make_cmd(0.05, 0.45)
        else:
            cmd = self.make_cmd(0.05, -0.45)

        self.pub.publish(cmd)
        self.mode_time -= 1


def main(args=None):
    rclpy.init(args=args)
    node = AutoCollectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    stop = TwistStamped()
    stop.header.stamp = node.get_clock().now().to_msg()
    stop.header.frame_id = 'base_link'
    stop.twist.linear.x = 0.0
    stop.twist.angular.z = 0.0
    node.pub.publish(stop)

    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
