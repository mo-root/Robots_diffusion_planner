#!/usr/bin/env python3
"""Baseline frontier explorer: heuristic scoring (no diffusion model).

Scores frontiers by: nearby unknown cells - distance cost.
This is the standard approach from Lecture 19 (Quin et al. 2014).
Used as a baseline to compare against our diffusion-based scorer.

Subscribes to /map, publishes /best_frontier.
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA


class BaselineFrontierNode(Node):
    def __init__(self):
        super().__init__("baseline_frontier_scorer")

        self.declare_parameter("score_interval", 5.0)
        self.declare_parameter("info_radius", 15)
        self.declare_parameter("beta_dist", 0.5)
        self.declare_parameter("min_frontier_size", 5)

        self.info_radius = self.get_parameter("info_radius").value
        self.beta_dist = self.get_parameter("beta_dist").value
        self.min_frontier_size = self.get_parameter("min_frontier_size").value

        self.map_sub = self.create_subscription(
            OccupancyGrid, "/map", self.map_callback, 10)
        self.frontier_pub = self.create_publisher(
            PoseStamped, "/best_frontier", 10)
        self.markers_pub = self.create_publisher(
            MarkerArray, "/scored_frontiers", 10)

        interval = self.get_parameter("score_interval").value
        self.timer = self.create_timer(interval, self.score_callback)

        self.grid = None
        self.map_info = None
        self.get_logger().info("BaselineFrontierScorer ready")

    def map_callback(self, msg: OccupancyGrid):
        w, h = msg.info.width, msg.info.height
        raw = np.array(msg.data, dtype=np.int8).reshape(h, w)
        self.grid = np.full((h, w), 0.5, dtype=np.float32)
        self.grid[raw == 0] = 1.0
        self.grid[raw == 100] = 0.0
        self.map_info = msg.info

    def score_callback(self):
        if self.grid is None:
            return

        known_mask = ((self.grid < 0.3) | (self.grid > 0.7)).astype(np.float32)
        h, w = known_mask.shape

        frontiers = []
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if known_mask[y, x] < 0.5:
                    continue
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and known_mask[ny, nx] < 0.5:
                        frontiers.append((y, x))
                        break

        if not frontiers:
            return

        frontier_set = set(frontiers)
        visited = set()
        clusters = []
        for f in frontiers:
            if f in visited:
                continue
            cluster = []
            stack = [f]
            while stack:
                cell = stack.pop()
                if cell in visited:
                    continue
                visited.add(cell)
                cluster.append(cell)
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                    n = (cell[0]+dy, cell[1]+dx)
                    if n in frontier_set and n not in visited:
                        stack.append(n)
            if len(cluster) >= self.min_frontier_size:
                cy = int(np.mean([c[0] for c in cluster]))
                cx = int(np.mean([c[1] for c in cluster]))
                clusters.append((cy, cx, len(cluster)))

        if not clusters:
            return

        robot_y, robot_x = h // 2, w // 2
        r = self.info_radius

        scored = []
        for cy, cx, size in clusters:
            y1, y2 = max(0, cy-r), min(h, cy+r+1)
            x1, x2 = max(0, cx-r), min(w, cx+r+1)
            unknown = (known_mask[y1:y2, x1:x2] < 0.5).sum()
            dist = math.sqrt((cy - robot_y)**2 + (cx - robot_x)**2)
            score = float(unknown) - self.beta_dist * dist
            scored.append((score, cy, cx))

        scored.sort(reverse=True)
        best_score, by, bx = scored[0]

        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        res = self.map_info.resolution
        msg.pose.position.x = self.map_info.origin.position.x + bx * res
        msg.pose.position.y = self.map_info.origin.position.y + by * res
        msg.pose.orientation.w = 1.0
        self.frontier_pub.publish(msg)

        self.get_logger().info(
            f"Baseline: {len(scored)} frontiers | "
            f"Best: unknown={best_score + self.beta_dist * math.sqrt((by-robot_y)**2 + (bx-robot_x)**2):.0f} "
            f"score={best_score:.1f}")


def main(args=None):
    rclpy.init(args=args)
    node = BaselineFrontierNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
