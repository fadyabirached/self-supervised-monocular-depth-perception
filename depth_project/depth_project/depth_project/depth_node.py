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

    def med(self, buf, val):
        buf.append(val)
        if len(buf) > self._N:
            buf.pop(0)
        return float(np.median(buf))

    def pct(self, arr, q):
        v = arr[np.isfinite(arr) & (arr > 0)]
        return float(np.percentile(v, q)) if len(v) > 0 else 0.0

    def image_callback(self, msg):
        frame = self.to_bgr(msg)
        img = self.transform(frame).unsqueeze(0).to(self.device)

        with torch.no_grad():
            disp, _ = self.model(img)
            depth = disp_to_depth(disp).squeeze().cpu().numpy()

        depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)

        h, w = depth.shape
        y1 = int(h * 0.25)
        y2 = int(h * 0.60)
        roi = depth[y1:y2, :]
        W = roi.shape[1]

        L_roi = roi[:, :int(W * 0.33)]
        C_roi = roi[:, int(W * 0.33):int(W * 0.67)]
        R_roi = roi[:, int(W * 0.67):]

        L = self.med(self._L, self.pct(L_roi.flatten(), 35))
        C = self.med(self._C, self.pct(C_roi.flatten(), 30))
        R = self.med(self._R, self.pct(R_roi.flatten(), 35))

        if C < min(L, R) * 0.85:
            steering = 0.0
        else:
            steering = -0.8 if R < L else 0.8

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
