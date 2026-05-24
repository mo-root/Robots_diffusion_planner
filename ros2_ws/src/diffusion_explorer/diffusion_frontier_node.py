#!/usr/bin/env python3
"""ROS 2 node: Diffusion-based frontier scorer for autonomous exploration.

Subscribes to /map (nav_msgs/OccupancyGrid) from PA4 mapper.
Publishes:
  /predicted_map (nav_msgs/OccupancyGrid) — best diffusion completion
  /scored_frontiers (visualization_msgs/MarkerArray) — scored frontier markers
  /best_frontier (geometry_msgs/PoseStamped) — goal for PA3 planner

Parameters:
  checkpoint_path: path to trained model .pt file
  device: 'cuda', 'mps', or 'cpu'
  K: number of diffusion samples (default 8)
  ddim_steps: DDIM sampling steps (default 50)
  lambda_var: variance bonus weight (default 1.0)
  beta_dist: distance penalty weight (default 0.5)
  info_radius: cells around frontier to count info gain (default 15)
  map_resolution: occupancy grid resolution in m/cell (default 0.05)
  score_interval: seconds between scoring updates (default 5.0)
  min_frontier_size: minimum cells to consider a frontier cluster (default 5)
"""

import sys
import os
import math
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
from unet import ConditionalUNet
from diffusion import DDPMScheduler


class DiffusionFrontierNode(Node):
    def __init__(self):
        super().__init__("diffusion_frontier_scorer")

        self.declare_parameter("checkpoint_path", "results/checkpoints/model_epoch0020.pt")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("K", 8)
        self.declare_parameter("ddim_steps", 50)
        self.declare_parameter("lambda_var", 1.0)
        self.declare_parameter("beta_dist", 0.5)
        self.declare_parameter("info_radius", 15)
        self.declare_parameter("score_interval", 5.0)
        self.declare_parameter("min_frontier_size", 5)
        self.declare_parameter("crop_size", 256)

        self.K = self.get_parameter("K").value
        self.ddim_steps = self.get_parameter("ddim_steps").value
        self.lambda_var = self.get_parameter("lambda_var").value
        self.beta_dist = self.get_parameter("beta_dist").value
        self.info_radius = self.get_parameter("info_radius").value
        self.min_frontier_size = self.get_parameter("min_frontier_size").value
        self.crop_size = self.get_parameter("crop_size").value

        self._load_model()

        self.map_sub = self.create_subscription(
            OccupancyGrid, "/map", self.map_callback, 10)

        self.predicted_map_pub = self.create_publisher(
            OccupancyGrid, "/predicted_map", 10)
        self.frontier_markers_pub = self.create_publisher(
            MarkerArray, "/scored_frontiers", 10)
        self.best_frontier_pub = self.create_publisher(
            PoseStamped, "/best_frontier", 10)

        interval = self.get_parameter("score_interval").value
        self.timer = self.create_timer(interval, self.score_callback)

        self.latest_map = None
        self.map_info = None
        self.robot_pos_grid = None

        self.get_logger().info(
            f"DiffusionFrontierScorer ready (K={self.K}, "
            f"ddim_steps={self.ddim_steps}, device={self.device})")

    def _load_model(self):
        checkpoint_path = self.get_parameter("checkpoint_path").value
        device_str = self.get_parameter("device").value
        self.device = torch.device(device_str)

        self.get_logger().info(f"Loading model from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        args = ckpt.get("args", {})

        self.model = ConditionalUNet(
            in_channels=3, out_channels=1,
            base_channels=args.get("base_channels", 32),
            channel_mults=tuple(args.get("channel_mults", [1, 2, 4, 4])),
            time_dim=args.get("time_dim", 128),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        self.scheduler = DDPMScheduler(
            num_timesteps=args.get("T", 1000), device=self.device)

        self.get_logger().info(
            f"Model loaded (epoch {ckpt.get('epoch', '?')}, "
            f"{sum(p.numel() for p in self.model.parameters()):,} params)")

    def map_callback(self, msg: OccupancyGrid):
        w, h = msg.info.width, msg.info.height
        raw = np.array(msg.data, dtype=np.int8).reshape(h, w)

        grid = np.full((h, w), 0.5, dtype=np.float32)
        grid[raw == 0] = 1.0    # free
        grid[raw == 100] = 0.0  # occupied
        # raw == -1 stays 0.5 (unknown)

        self.latest_map = grid
        self.map_info = msg.info

        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y
        res = msg.info.resolution
        self.robot_pos_grid = (
            int(-origin_y / res + h / 2),
            int(-origin_x / res + w / 2)
        )

    def _crop_around_robot(self, grid):
        """Crop/pad grid to crop_size x crop_size centered on robot."""
        h, w = grid.shape
        cs = self.crop_size

        if self.robot_pos_grid:
            cy, cx = self.robot_pos_grid
        else:
            cy, cx = h // 2, w // 2

        y1 = max(0, cy - cs // 2)
        y2 = min(h, cy + cs // 2)
        x1 = max(0, cx - cs // 2)
        x2 = min(w, cx + cs // 2)

        cropped = np.full((cs, cs), 0.5, dtype=np.float32)
        dy = (cs - (y2 - y1)) // 2
        dx = (cs - (x2 - x1)) // 2
        cropped[dy:dy + (y2 - y1), dx:dx + (x2 - x1)] = grid[y1:y2, x1:x2]

        offset = (y1 - dy, x1 - dx)
        return cropped, offset

    def detect_frontiers(self, known_mask):
        """Find frontier cells on the boundary of known and unknown."""
        h, w = known_mask.shape
        frontiers = []
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if known_mask[y, x] < 0.5:
                    continue
                unknown_neighbors = 0
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and known_mask[ny, nx] < 0.5:
                        unknown_neighbors += 1
                if unknown_neighbors >= 1:
                    frontiers.append((y, x))
        return frontiers

    def cluster_frontiers(self, frontiers):
        """Group adjacent frontier cells into clusters, return centroids."""
        if not frontiers:
            return []

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
                    neighbor = (cell[0]+dy, cell[1]+dx)
                    if neighbor in frontier_set and neighbor not in visited:
                        stack.append(neighbor)
            if len(cluster) >= self.min_frontier_size:
                cy = int(np.mean([c[0] for c in cluster]))
                cx = int(np.mean([c[1] for c in cluster]))
                clusters.append((cy, cx, len(cluster)))

        return clusters

    @torch.no_grad()
    def score_callback(self):
        if self.latest_map is None:
            return

        grid = self.latest_map
        known_mask = ((grid < 0.3) | (grid > 0.7)).astype(np.float32)

        coverage = known_mask.mean()
        if coverage > 0.95:
            self.get_logger().info("Map >95% explored. Exploration complete.")
            return

        cropped, offset = self._crop_around_robot(grid)
        cropped_mask = ((cropped < 0.3) | (cropped > 0.7)).astype(np.float32)

        pm = torch.tensor(cropped * 2.0 - 1.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        km = torch.tensor(cropped_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        completions = []
        for _ in range(self.K):
            pred = self.scheduler.sample_ddim(self.model, pm, km, num_steps=self.ddim_steps)
            pred_np = (pred[0, 0].cpu().numpy() + 1) / 2
            completions.append(pred_np.clip(0, 1))

        mean_completion = np.mean(completions, axis=0)
        self._publish_predicted_map(mean_completion, offset)

        frontiers = self.detect_frontiers(cropped_mask)
        clusters = self.cluster_frontiers(frontiers)

        if not clusters:
            self.get_logger().info("No frontiers found.")
            return

        robot_y = self.crop_size // 2
        robot_x = self.crop_size // 2

        scored = []
        for cy, cx, size in clusters:
            gains = []
            for comp in completions:
                r = self.info_radius
                y1, y2 = max(0, cy-r), min(self.crop_size, cy+r+1)
                x1, x2 = max(0, cx-r), min(self.crop_size, cx+r+1)
                region = comp[y1:y2, x1:x2]
                mask_region = cropped_mask[y1:y2, x1:x2]
                gain = float(((region > 0.5) & (mask_region < 0.5)).sum())
                gains.append(gain)

            expected = np.mean(gains)
            variance = np.std(gains)
            dist = math.sqrt((cy - robot_y)**2 + (cx - robot_x)**2)

            score = expected + self.lambda_var * variance - self.beta_dist * dist
            scored.append((score, cy, cx, size, expected, variance, dist))

        scored.sort(reverse=True)
        self._publish_frontier_markers(scored, offset)

        best_score, by, bx, bsize, bexp, bvar, bdist = scored[0]
        self._publish_best_frontier(by + offset[0], bx + offset[1])

        self.get_logger().info(
            f"Scored {len(scored)} frontiers | "
            f"Best: gain={bexp:.0f} var={bvar:.1f} dist={bdist:.0f} "
            f"score={best_score:.1f} | coverage={coverage:.0%}")

    def _publish_predicted_map(self, pred, offset):
        msg = OccupancyGrid()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        msg.info = MapMetaData()
        msg.info.resolution = self.map_info.resolution if self.map_info else 0.05
        msg.info.width = self.crop_size
        msg.info.height = self.crop_size

        if self.map_info:
            msg.info.origin = self.map_info.origin
            msg.info.origin.position.x += offset[1] * msg.info.resolution
            msg.info.origin.position.y += offset[0] * msg.info.resolution

        data = np.zeros(self.crop_size * self.crop_size, dtype=np.int8)
        flat = pred.flatten()
        data[flat > 0.5] = 0      # free
        data[flat <= 0.5] = 100   # occupied
        msg.data = data.tolist()

        self.predicted_map_pub.publish(msg)

    def _publish_frontier_markers(self, scored, offset):
        markers = MarkerArray()
        res = self.map_info.resolution if self.map_info else 0.05
        ox = self.map_info.origin.position.x if self.map_info else 0.0
        oy = self.map_info.origin.position.y if self.map_info else 0.0

        max_score = scored[0][0] if scored else 1.0
        min_score = scored[-1][0] if scored else 0.0
        score_range = max(max_score - min_score, 1e-6)

        for i, (score, cy, cx, size, exp, var, dist) in enumerate(scored):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "frontiers"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD

            m.pose.position.x = ox + (cx + offset[1]) * res
            m.pose.position.y = oy + (cy + offset[0]) * res
            m.pose.position.z = 0.1
            m.pose.orientation.w = 1.0

            scale = 0.2 + 0.5 * (score - min_score) / score_range
            m.scale.x = scale
            m.scale.y = scale
            m.scale.z = scale

            ratio = (score - min_score) / score_range
            m.color = ColorRGBA(
                r=1.0 - ratio, g=ratio, b=0.0, a=0.8)

            m.lifetime.sec = 5
            markers.markers.append(m)

        self.frontier_markers_pub.publish(markers)

    def _publish_best_frontier(self, grid_y, grid_x):
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()

        res = self.map_info.resolution if self.map_info else 0.05
        ox = self.map_info.origin.position.x if self.map_info else 0.0
        oy = self.map_info.origin.position.y if self.map_info else 0.0

        msg.pose.position.x = ox + grid_x * res
        msg.pose.position.y = oy + grid_y * res
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0

        self.best_frontier_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DiffusionFrontierNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
