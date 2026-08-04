import os
import cv2
import numpy as np
import torch
import rclpy
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from depth_project.models.depth_net import DepthNet
from depth_project.infer_image import preprocess
from depth_project.losses import disp_to_depth
from depth_project.steering_logic import split_regions, percentile_of_valid, median_filter, choose_steering


def find_checkpoint():
    """Locate the trained weights without assuming a workspace layout.

    This used to be a single hardcoded ~/ros2_ws/src path, which is one of
    the absolute-path assumptions listed in the README's known limitations
    and the reason the node could not start from a container or from any
    workspace not at that exact location. Checked in order: an explicit
    override, the installed package share, then the legacy path so an
    existing ~/ros2_ws setup keeps working unchanged.
    """
    candidates = []

    override = os.environ.get('DEPTH_CHECKPOINT')
    if override:
        candidates.append(os.path.expanduser(override))

    try:
        from ament_index_python.packages import get_package_share_directory
        candidates.append(os.path.join(
            get_package_share_directory('depth_project'),
            'checkpoints', 'selfsup_depth_latest.pth'))
    except Exception:  # package not installed, e.g. running from source
        pass

    candidates.append(os.path.expanduser(
        '~/ros2_ws/src/depth_project/checkpoints/selfsup_depth_latest.pth'))

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        'checkpoint not found, looked in:\n  ' + '\n  '.join(candidates)
        + '\nSet DEPTH_CHECKPOINT to point at selfsup_depth_latest.pth.')


class DepthNode(Node):
    def __init__(self):
        super().__init__('depth_node')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DepthNet().to(self.device)

        # No display means no cv2.imshow: the node runs headless (in a
        # container, or on a server) and just publishes /steering_cmd.
        self.show_window = bool(os.environ.get('DISPLAY'))

        ckpt = find_checkpoint()
        self.get_logger().info(f'Loading checkpoint: {ckpt}')

        data = torch.load(ckpt, map_location=self.device)
        self.model.load_state_dict(data['depth_net'])
        self.model.eval()

        self.sub = self.create_subscription(Image, '/camera', self.image_callback, 10)
        self.pub = self.create_publisher(Float32, '/steering_cmd', 10)

        self._N = 6
        self._L, self._C, self._R = [], [], []
        self.get_logger().info('Self-supervised depth node ready.')

    def to_rgb(self, msg):
        """Decode a sensor_msgs/Image into an RGB array for the model.

        RGB, not BGR. DepthNet was trained on PIL images opened with
        .convert('RGB') (see dataset_sequence.SequenceDataset), so handing
        it a BGR frame here would be a train/serve mismatch: the network
        would be scoring channel statistics it never learned. Measured on
        coloured frames the two orderings disagree by roughly half the mean
        predicted depth, enough to flip the published steering command.
        OpenCV still wants BGR, so the display path converts back below.
        """
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        if msg.encoding == 'bgr8':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def image_callback(self, msg):
        frame = self.to_rgb(msg)
        img = preprocess(PILImage.fromarray(frame)).to(self.device)

        with torch.no_grad():
            disp, _ = self.model(img)
            depth = disp_to_depth(disp).squeeze().cpu().numpy()

        L_roi, C_roi, R_roi = split_regions(depth)

        L = median_filter(self._L, percentile_of_valid(L_roi, 35), self._N)
        C = median_filter(self._C, percentile_of_valid(C_roi, 30), self._N)
        R = median_filter(self._R, percentile_of_valid(R_roi, 35), self._N)

        steering = choose_steering(L, C, R)

        out = Float32()
        out.data = float(np.clip(steering, -1.0, 1.0))
        self.pub.publish(out)

        if not self.show_window:
            return

        depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)

        bgr_vis = cv2.cvtColor(cv2.resize(frame, (320, 192)), cv2.COLOR_RGB2BGR)
        combined = np.hstack((bgr_vis, depth_vis))
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
