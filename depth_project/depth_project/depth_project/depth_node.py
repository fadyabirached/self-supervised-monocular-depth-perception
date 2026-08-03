import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from depth_project.models.depth_net import DepthNet
from depth_project.losses import disp_to_depth
from depth_project.steering_logic import split_regions, percentile_of_valid, median_filter, choose_steering


class DepthNode(Node):
    def __init__(self):
        super().__init__('depth_node')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DepthNet().to(self.device)

        ckpt = os.path.expanduser('~/ros2_ws/src/depth_project/checkpoints/selfsup_depth_latest.pth')
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f'checkpoint not found: {ckpt}')

        data = torch.load(ckpt, map_location=self.device)
        self.model.load_state_dict(data['depth_net'])
        self.model.eval()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((192, 320)),
            T.ToTensor(),
        ])

        self.sub = self.create_subscription(Image, '/camera', self.image_callback, 10)
        self.pub = self.create_publisher(Float32, '/steering_cmd', 10)

        self._N = 6
        self._L, self._C, self._R = [], [], []
        self.get_logger().info('Self-supervised depth node ready.')

    def to_bgr(self, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        if msg.encoding == 'rgb8':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def image_callback(self, msg):
        frame = self.to_bgr(msg)
        img = self.transform(frame).unsqueeze(0).to(self.device)

        with torch.no_grad():
            disp, _ = self.model(img)
            depth = disp_to_depth(disp).squeeze().cpu().numpy()

        depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)

        L_roi, C_roi, R_roi = split_regions(depth)

        L = median_filter(self._L, percentile_of_valid(L_roi, 35), self._N)
        C = median_filter(self._C, percentile_of_valid(C_roi, 30), self._N)
        R = median_filter(self._R, percentile_of_valid(R_roi, 35), self._N)

        steering = choose_steering(L, C, R)

        out = Float32()
        out.data = float(np.clip(steering, -1.0, 1.0))
        self.pub.publish(out)

        rgb_vis = cv2.resize(frame, (320, 192))
        combined = np.hstack((rgb_vis, depth_vis))
        cv2.putText(combined, 'RGB', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(combined, 'SELF-SUP DEPTH', (340, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow('RGB | Self-Supervised Depth', combined)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = DepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
