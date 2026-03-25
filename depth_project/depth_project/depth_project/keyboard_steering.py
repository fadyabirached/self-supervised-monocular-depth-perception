import sys
import termios
import tty
import select

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32


MSG = '''
Keyboard control:
  w : forward (straight)
  a : turn left
  d : turn right
  s : stop turning / straight
  x : stop robot
  q : quit
'''


class KeyboardSteering(Node):
    def __init__(self):
        super().__init__('keyboard_steering')
        self.pub = self.create_publisher(Float32, '/steering_cmd', 10)
        self.current = 0.0
        self.get_logger().info('Keyboard steering started')
        print(MSG)

    def publish_value(self, value: float):
        msg = Float32()
        msg.data = float(value)
        self.pub.publish(msg)
        self.get_logger().info(f'steering_cmd = {msg.data:.2f}')

    def get_key(self):
        dr, _, _ = select.select([sys.stdin], [], [], 0.1)
        if dr:
            return sys.stdin.read(1)
        return ''

    def run(self):
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while rclpy.ok():
                key = self.get_key()

                if key == 'w':
                    self.current = 0.0
                    self.publish_value(self.current)

                elif key == 'a':
                    self.current = 0.8
                    self.publish_value(self.current)

                elif key == 'd':
                    self.current = -0.8
                    self.publish_value(self.current)

                elif key == 's':
                    self.current = 0.0
                    self.publish_value(self.current)

                elif key == 'x':
                    self.current = 0.0
                    self.publish_value(self.current)

                elif key == 'q':
                    self.current = 0.0
                    self.publish_value(self.current)
                    break

                rclpy.spin_once(self, timeout_sec=0.01)

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardSteering()
    node.run()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
