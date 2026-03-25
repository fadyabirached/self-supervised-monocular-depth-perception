import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool


class SaveImagesNode(Node):
    def __init__(self):
        super().__init__('save_images_node')

        self.sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_cb,
            10
        )

        self.trigger_sub = self.create_subscription(
            Bool,
            '/capture_image',
            self.capture_cb,
            10
        )

        self.latest_image = None
        self.count = 0
        self.out_dir = os.path.expanduser('~/yolo_dataset/raw')
        os.makedirs(self.out_dir, exist_ok=True)

        self.get_logger().info(f"Saving images to {self.out_dir}")

    def image_cb(self, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        if msg.encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.latest_image = img

    def capture_cb(self, msg: Bool):
        if not msg.data:
            return

        if self.latest_image is None:
            self.get_logger().warn("No image available yet.")
            return

        path = os.path.join(self.out_dir, f"frame_{self.count:06d}.jpg")
        cv2.imwrite(path, self.latest_image)
        self.get_logger().info(f"Saved {path}")
        self.count += 1


def main(args=None):
    rclpy.init(args=args)
    node = SaveImagesNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
