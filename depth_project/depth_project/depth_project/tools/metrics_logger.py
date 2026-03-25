#!/usr/bin/env python3
import csv
import os
import time
import threading
from dataclasses import dataclass, asdict

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32


@dataclass
class RunMetrics:
    run_id: int
    method: str
    obstacle_time: float = 0.0
    reaction_time: float = -1.0
    reacted: int = 0
    success: int = 0
    collision: int = 0
    steering_at_reaction: float = 0.0


class MetricsLogger(Node):
    def __init__(self):
        super().__init__('metrics_logger')

        self.declare_parameter('method', 'depth')
        self.declare_parameter('csv_path', os.path.expanduser('~/robot_metrics_log.csv'))
        self.declare_parameter('steering_threshold', 0.01)

        self.method = str(self.get_parameter('method').value)
        self.csv_path = str(self.get_parameter('csv_path').value)
        self.steering_threshold = float(self.get_parameter('steering_threshold').value)

        self.run_id = 1
        self.current = RunMetrics(run_id=self.run_id, method=self.method)

        self.latest_steering = 0.0
        self.obstacle_marked = False
        self.lock = threading.Lock()

        self.create_subscription(Float32, '/steering_cmd', self.steering_cb, 10)

        self._ensure_csv_header()

        self.get_logger().info(f'Method: {self.method}')
        self.get_logger().info(f'CSV: {self.csv_path}')
        self.get_logger().info('Controls:')
        self.get_logger().info('  o = obstacle appeared')
        self.get_logger().info('  c = collision')
        self.get_logger().info('  s = success')
        self.get_logger().info('  n = next run')
        self.get_logger().info('  q = quit')

    def _ensure_csv_header(self):
        os.makedirs(os.path.dirname(self.csv_path) or '.', exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(asdict(self.current).keys()))
                writer.writeheader()

    def steering_cb(self, msg):
        with self.lock:
            self.latest_steering = float(msg.data)

            if self.obstacle_marked and not self.current.reacted:
                if abs(self.latest_steering) >= self.steering_threshold:
                    now = time.time()
                    self.current.reaction_time = now - self.current.obstacle_time
                    self.current.reacted = 1
                    self.current.steering_at_reaction = self.latest_steering
                    self.get_logger().info(
                        f'Reaction detected: {self.current.reaction_time:.3f}s, steering={self.latest_steering:.3f}'
                    )

    def mark_obstacle(self):
        with self.lock:
            self.current.obstacle_time = time.time()
            self.current.reaction_time = -1.0
            self.current.reacted = 0
            self.current.success = 0
            self.current.collision = 0
            self.current.steering_at_reaction = 0.0
            self.obstacle_marked = True
            self.get_logger().info(f'Obstacle marked for run {self.run_id}')

    def mark_collision(self):
        with self.lock:
            self.current.collision = 1
            self.current.success = 0
            self._save_current()
            self._next_run()
            self.get_logger().info('Collision saved')

    def mark_success(self):
        with self.lock:
            self.current.success = 1
            self.current.collision = 0
            self._save_current()
            self._next_run()
            self.get_logger().info('Success saved')

    def next_run(self):
        with self.lock:
            self._save_current()
            self._next_run()
            self.get_logger().info('Moved to next run')

    def _save_current(self):
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(self.current).keys()))
            writer.writerow(asdict(self.current))

    def _next_run(self):
        self.run_id += 1
        self.current = RunMetrics(run_id=self.run_id, method=self.method)
        self.latest_steering = 0.0
        self.obstacle_marked = False


def keyboard_thread(node):
    while rclpy.ok():
        try:
            cmd = input().strip().lower()
        except EOFError:
            break

        if cmd == 'o':
            node.mark_obstacle()
        elif cmd == 'c':
            node.mark_collision()
        elif cmd == 's':
            node.mark_success()
        elif cmd == 'n':
            node.next_run()
        elif cmd == 'q':
            rclpy.shutdown()
            break
        elif cmd:
            print('Use: o, c, s, n, q')


def main():
    rclpy.init()
    node = MetricsLogger()

    t = threading.Thread(target=keyboard_thread, args=(node,), daemon=True)
    t.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
