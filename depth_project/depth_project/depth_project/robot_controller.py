import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import TwistStamped


class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')

        self.subscription = self.create_subscription(
            Float32,
            '/steering_cmd',
            self.steering_callback,
            10
        )

        self.cmd_pub = self.create_publisher(
            TwistStamped,
            '/model/waffle/cmd_vel',
            10
        )

        self.current_steering = 0.0
        self.timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info('Robot controller ready.')

    def steering_callback(self, msg):
        self.current_steering = float(msg.data)

    def control_loop(self):
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'
        cmd.twist.linear.x = 0.20
        cmd.twist.angular.z = 1.8 * self.current_steering
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
