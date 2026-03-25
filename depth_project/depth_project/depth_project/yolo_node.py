import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
from ultralytics import YOLO


class YoloNavigator(Node):
    def __init__(self):
        super().__init__('yolo_navigator')

        self.bridge = CvBridge()
        self.model = YOLO('yolov8n.pt')

        self.image_sub = self.create_subscription(
            Image,
            '/camera',
            self.image_callback,
            10
        )

        self.steering_pub = self.create_publisher(
            Float32,
            '/steering_cmd',
            10
        )

        self.debug_pub = self.create_publisher(
            Image,
            '/yolo_debug',
            10
        )

        self.target_classes = {'chair'}
        self.forward_steering = 0.0
        self.turn_left = 0.8
        self.turn_right = -0.8

        self.get_logger().info('YOLO navigator started. Subscribing to /camera')

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CV bridge error: {e}')
            return

        _, w, _ = frame.shape
        image_center_x = w / 2.0

        results = self.model(frame, verbose=False)

        best_target = None
        best_area = 0.0

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                class_name = self.model.names[cls_id]

                if class_name not in self.target_classes:
                    continue
                if conf < 0.20:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                area = (x2 - x1) * (y2 - y1)
                center_x = (x1 + x2) / 2.0

                self.get_logger().info(
                    f'detected {class_name} conf={conf:.2f} area={area:.1f}'
                )

                if area > best_area:
                    best_area = area
                    best_target = {'center_x': center_x, 'area': area}

        steering = Float32()

        if best_target is None:
            steering.data = self.forward_steering
        else:
            offset = best_target['center_x'] - image_center_x
            area = best_target['area']

            center_threshold = 40.0
            close_area = 15000.0

            if area > close_area:
                steering.data = self.turn_left
            else:
                if offset < -center_threshold:
                    steering.data = self.turn_right
                elif offset > center_threshold:
                    steering.data = self.turn_left
                else:
                    steering.data = self.forward_steering

        self.steering_pub.publish(steering)

        try:
            debug_img = results[0].plot() if len(results) > 0 else frame
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
            self.debug_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f'Debug image publish error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = YoloNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
