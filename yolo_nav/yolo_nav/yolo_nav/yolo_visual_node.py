import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2


class YoloVisualizer(Node):
    def __init__(self):
        super().__init__('yolo_visualizer')

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

        self.get_logger().info('YOLO visualizer started')

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(str(e))
            return

        results = self.model(frame, verbose=False)
        annotated = frame.copy()

        steering = Float32()
        steering.data = 0.0

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = self.model.names[cls]

                cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(annotated, name, (x1,y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        cv2.imshow("YOLO", annotated)
        cv2.waitKey(1)

        self.steering_pub.publish(steering)


def main(args=None):
    rclpy.init(args=args)
    node = YoloVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
