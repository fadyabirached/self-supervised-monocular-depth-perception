import os
import json
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo


class CollectSelfSup(Node):
    def __init__(self):
        super().__init__('collect_selfsup')

        self.out_dir = os.path.expanduser('~/depth_selfsup_data')
        os.makedirs(self.out_dir, exist_ok=True)

        self.count = 0
        self.latest = None
        self.saved_info = False

        self.create_subscription(Image, '/camera', self.image_cb, 10)
        self.create_subscription(CameraInfo, '/camera/camera_info', self.info_cb, 10)
        self.timer = self.create_timer(0.5, self.save_cb)

        self.get_logger().info(f'Collecting to {self.out_dir}')

    def image_cb(self, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        if msg.encoding == 'rgb8':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.latest = img

    def info_cb(self, msg):
        if self.saved_info:
            return
        info = {
            'width': msg.width,
            'height': msg.height,
            'k': list(msg.k),
        }
        with open(os.path.join(self.out_dir, 'camera_info.json'), 'w') as f:
            json.dump(info, f, indent=2)
        self.saved_info = True
        self.get_logger().info('Saved camera_info.json')

    def save_cb(self):
        if self.latest is None:
            return
        path = os.path.join(self.out_dir, f'frame_{self.count:06d}.png')
        cv2.imwrite(path, self.latest)
        self.get_logger().info(f'Saved {path}')
        self.count += 1


def main(args=None):
    rclpy.init(args=args)
    node = CollectSelfSup()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
