import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2


class YoloNavigator(Node):
    def __init__(self):
        super().__init__('yolo_navigator')
        self.bridge = CvBridge()
        self.get_logger().info('Loading YOLOv8n model...')
        self.model = YOLO('yolov8n.pt')
        self.get_logger().info('Model loaded.')

        self.image_sub = self.create_subscription(
            Image, '/camera', self.image_callback, 10)

        self.steering_pub = self.create_publisher(Float32, '/steering_cmd', 10)
        self.speed_pub    = self.create_publisher(Float32, '/speed_cmd',    10)

        self.target_classes   = {'chair'}
        self.close_area       = 15000.0
        self.center_threshold = 40.0
        self.get_logger().info('YOLO navigator ready — subscribing to /camera')

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CV bridge error: {e}')
            return

        h, w = frame.shape[:2]
        image_center_x = w / 2.0
        results = self.model(frame, verbose=False)

        best_target = None
        best_area   = 0.0
        vis = frame.copy()

        cv2.line(vis, (int(w * 0.4), 0), (int(w * 0.4), h), (255, 255, 0), 1)
        cv2.line(vis, (int(w * 0.6), 0), (int(w * 0.6), h), (255, 255, 0), 1)

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id     = int(box.cls[0].item())
                conf       = float(box.conf[0].item())
                class_name = self.model.names[cls_id]
                if class_name not in self.target_classes or conf < 0.20:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                area     = (x2 - x1) * (y2 - y1)
                center_x = (x1 + x2) / 2.0
                self.get_logger().info(
                    f'[DETECTED] {class_name}  conf={conf:.2f}  area={area:.0f}  cx={center_x:.0f}')
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f'{class_name} {conf:.2f}', (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.circle(vis, (int(center_x), (y1 + y2) // 2), 5, (0, 0, 255), -1)
                if area > best_area:
                    best_area   = area
                    best_target = {'center_x': center_x, 'area': area}

        steering = Float32()
        speed    = Float32()

        if best_target is None:
            self.get_logger().info('[NAV] No chair — spinning to search')
            steering.data = 0.5
            speed.data    = 0.0
            status_text   = 'SEARCHING...'
            status_color  = (0, 165, 255)
        else:
            offset = best_target['center_x'] - image_center_x
            area   = best_target['area']
            if area > self.close_area:
                steering.data = 0.8
                speed.data    = 0.0
                status_text   = 'TOO CLOSE — TURNING'
                status_color  = (0, 0, 255)
                self.get_logger().info('[NAV] Too close — turning')
            elif offset < -self.center_threshold:
                steering.data = -0.6
                speed.data    = 0.10
                status_text   = f'TURN LEFT  offset={offset:.0f}'
                status_color  = (255, 100, 0)
                self.get_logger().info(f'[NAV] Chair LEFT offset={offset:.0f}')
            elif offset > self.center_threshold:
                steering.data = 0.6
                speed.data    = 0.10
                status_text   = f'TURN RIGHT  offset={offset:.0f}'
                status_color  = (255, 100, 0)
                self.get_logger().info(f'[NAV] Chair RIGHT offset={offset:.0f}')
            else:
                steering.data = 0.0
                speed.data    = 0.15
                status_text   = f'FORWARD  area={area:.0f}'
                status_color  = (0, 255, 0)
                self.get_logger().info(f'[NAV] Centered — forward  area={area:.0f}')

        self.steering_pub.publish(steering)
        self.speed_pub.publish(speed)

        cv2.putText(vis, 'YOLO Chair Navigator', (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(vis, status_text, (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(vis,
                    f'speed={speed.data:.2f}  steer={steering.data:.2f}',
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.imshow('Robot Vision — YOLO Chair Detection', vis)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = YoloNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
