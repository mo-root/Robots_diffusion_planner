#!/usr/bin/env python3
"""PA-4 - Occupancy Grid Mapping (from-scratch).

Implements the deliverable required by the PA-4 rubric:

  Mapping engine        : recursive Bayesian occupancy grid backed by a log-odds
                          buffer.  Each cell stores a real-valued log-odds value
                          and is exposed on /map as the standard
                          nav_msgs/OccupancyGrid (-1 / 0 / 100).

  Raycasting            : Bresenham's integer-only line algorithm walks every
                          laser beam from the robot cell to the endpoint cell,
                          marking intermediate cells as free evidence and the
                          endpoint as occupied evidence (only when the reading
                          is strictly below max-range -> otherwise the entire
                          ray is "free up to max range").

  Frame handling        : every laser hit is transformed laser_frame ->
                          base_link -> odom via tf2 lookups.  The grid lives in
                          the odom frame (pose assumed trustworthy as discussed
                          in lec16).

  Map growth            : if a hit / endpoint cell would land outside the
                          current grid, the underlying log-odds buffer is
                          re-allocated with extra padding on the offending
                          side(s) and the existing data is copied in.  The
                          published OccupancyGrid origin shifts accordingly
                          so RViz lines up.

Extra-credit extensions (both implemented):

  Log-odds update       : default mode.  Cell log-odds is incremented by
                          L_FREE on a free observation and L_OCC on a hit;
                          values are clamped to [L_MIN, L_MAX] to keep the
                          filter responsive to dynamic obstacles.

  Binary fallback       : pass `update_mode:=binary` to use a hard 0/100
                          assignment instead -- closer to the base-credit
                          spec.

Author: Moin Mattar - COSC 81/281, Spring 2026.
AI helped me with code formatting and Python style.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.time import Time

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, Quaternion

import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


# =============================================================================
# Tuning constants (also exposed as ROS parameters where it makes sense)
# =============================================================================
# Cell size in metres.  Smaller -> sharper map, but quadratically more memory
# and per-scan work.  Default 0.05 m matches typical SLAM tutorials.
DEFAULT_RESOLUTION = 0.05

# Initial map size around the robot's first observed pose.  The buffer grows
# automatically as the robot explores, so this is just a starting allocation.
DEFAULT_INIT_SIZE = 12.0  # metres on a side

# Padding when growing the grid - at least this many cells of buffer are
# allocated past the new extreme so we don't reallocate every other frame.
GROWTH_PAD_CELLS = 32

# Log-odds increments.  Tuned so a single hit shifts the cell roughly 0.85,
# and a single free observation roughly -0.4 (free evidence is weaker because
# Bresenham crosses many free cells per scan -- otherwise the map would
# saturate to "free" after one frame).
DEFAULT_L_OCC = +0.85
DEFAULT_L_FREE = -0.40
L_MIN = -8.0
L_MAX = +8.0
# Prior log-odds for an unknown cell (P=0.5 -> log-odds = 0).
L_PRIOR = 0.0

# Decision threshold turning log-odds into occupied/free for the published map.
# 50/50 prior, hysteresis bands so unknown cells stay "unknown" instead of
# flickering between 0 and 100.
OCC_THRESHOLD = 0.85   # > this -> occupied (100)
FREE_THRESHOLD = -0.40  # < this -> free (0)

# Process every Nth scan to keep CPU manageable.  N=1 -> every scan.
DEFAULT_SCAN_STRIDE = 1

# Throw away rays that come back NaN/inf or below this minimum range.
RANGE_MIN_FALLBACK = 0.05


# =============================================================================
# Bresenham line algorithm (integer cells, 8-octant).
# =============================================================================
def bresenham_line(c0: int, r0: int, c1: int, r1: int) -> List[Tuple[int, int]]:
    """Return the list of (col, row) cells from (c0, r0) to (c1, r1) inclusive.

    Bresenham's classic integer-only rasterisation.  Handles all 8 octants by
    flipping signs / swapping axes so that we always advance the "fast" axis
    by 1 per step and the "slow" axis by an error-driven 0/1 increment.

    Reference: lec16 transcript - prof writes the algorithm assuming
    0 <= m <= 1 and notes the other quadrants need symmetric handling.
    """
    cells: List[Tuple[int, int]] = []
    dc = abs(c1 - c0)
    dr = -abs(r1 - r0)
    sc = 1 if c0 < c1 else -1
    sr = 1 if r0 < r1 else -1
    err = dc + dr
    c, r = c0, r0
    while True:
        cells.append((c, r))
        if c == c1 and r == r1:
            return cells
        e2 = 2 * err
        if e2 >= dr:
            err += dr
            c += sc
        if e2 <= dc:
            err += dc
            r += sr


# =============================================================================
# Pose helpers
# =============================================================================
@dataclass
class Pose2D:
    x: float
    y: float
    theta: float


def quat_to_yaw(q: Quaternion) -> float:
    """Yaw from a quaternion (z-axis rotation only)."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def transform_to_pose2d(t) -> Pose2D:
    """Extract a 2-D pose from a TransformStamped."""
    x = t.transform.translation.x
    y = t.transform.translation.y
    theta = quat_to_yaw(t.transform.rotation)
    return Pose2D(x, y, theta)


# =============================================================================
# Log-odds occupancy grid with dynamic growth.
# =============================================================================
class LogOddsGrid:
    """2-D occupancy grid stored as log-odds floats.

    Coordinate system:
      - World/odom metres: (wx, wy).
      - Grid indices: (col, row) with col -> +x, row -> +y.
      - Cell (0, 0) is at world position (origin_x, origin_y) (its lower-left
        corner).  This matches nav_msgs/OccupancyGrid origin semantics.

    The internal buffer is a flat row-major list of floats, length
    width * height.  Conversion to / from a published OccupancyGrid happens in
    `to_msg`.
    """

    def __init__(
        self,
        width: int,
        height: int,
        resolution: float,
        origin_x: float,
        origin_y: float,
    ) -> None:
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.data: List[float] = [L_PRIOR] * (width * height)

    # --- world <-> grid conversions ---
    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        c = int(math.floor((wx - self.origin_x) / self.resolution))
        r = int(math.floor((wy - self.origin_y) / self.resolution))
        return c, r

    def in_bounds(self, c: int, r: int) -> bool:
        return 0 <= c < self.width and 0 <= r < self.height

    def cell_idx(self, c: int, r: int) -> int:
        return r * self.width + c

    # --- log-odds bookkeeping ---
    def update(self, c: int, r: int, delta: float) -> None:
        """Add `delta` to the cell's log-odds, clamped to [L_MIN, L_MAX]."""
        if not self.in_bounds(c, r):
            return
        idx = self.cell_idx(c, r)
        v = self.data[idx] + delta
        if v < L_MIN:
            v = L_MIN
        elif v > L_MAX:
            v = L_MAX
        self.data[idx] = v

    def set_binary(self, c: int, r: int, occupied: bool) -> None:
        """Hard 0/100 assignment for the binary-update fallback mode."""
        if not self.in_bounds(c, r):
            return
        self.data[self.cell_idx(c, r)] = L_MAX if occupied else L_MIN

    # --- growth ---
    def ensure_contains(self, c: int, r: int) -> None:
        """If (c, r) lies outside the grid, reallocate with padding.

        The published origin shifts by the same amount we add to the bottom /
        left side so RViz keeps showing a continuous map.
        """
        pad_left = max(0, -c + GROWTH_PAD_CELLS) if c < 0 else 0
        pad_right = max(0, c - (self.width - 1) + GROWTH_PAD_CELLS) if c >= self.width else 0
        pad_bottom = max(0, -r + GROWTH_PAD_CELLS) if r < 0 else 0
        pad_top = max(0, r - (self.height - 1) + GROWTH_PAD_CELLS) if r >= self.height else 0

        if pad_left == pad_right == pad_bottom == pad_top == 0:
            return

        new_w = self.width + pad_left + pad_right
        new_h = self.height + pad_bottom + pad_top
        new_data: List[float] = [L_PRIOR] * (new_w * new_h)

        # Copy old buffer into the offset window of the new buffer.
        for old_r in range(self.height):
            new_r = old_r + pad_bottom
            old_off = old_r * self.width
            new_off = new_r * new_w + pad_left
            for old_c in range(self.width):
                new_data[new_off + old_c] = self.data[old_off + old_c]

        self.data = new_data
        self.width = new_w
        self.height = new_h
        self.origin_x -= pad_left * self.resolution
        self.origin_y -= pad_bottom * self.resolution

    # --- export ---
    def to_msg(self, frame_id: str, stamp) -> OccupancyGrid:
        msg = OccupancyGrid()
        msg.header = Header()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.info.resolution = self.resolution
        msg.info.width = self.width
        msg.info.height = self.height
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0  # identity

        # Convert log-odds floats to int8 with hysteresis around the prior.
        out: List[int] = []
        for v in self.data:
            if v >= OCC_THRESHOLD:
                out.append(100)
            elif v <= FREE_THRESHOLD:
                out.append(0)
            elif abs(v - L_PRIOR) < 1e-9:
                out.append(-1)  # truly untouched
            else:
                # In the band -- linear interpolate so RViz's heatmap still
                # shows accumulating evidence.
                p = 1.0 / (1.0 + math.exp(-v))
                out.append(int(round(100.0 * p)))
        msg.data = out
        return msg


# =============================================================================
# ROS 2 node
# =============================================================================
class GridMapperNode(Node):
    def __init__(self) -> None:
        super().__init__("pa4_grid_mapper")

        # ---- Parameters ---------------------------------------------------
        self.declare_parameter("resolution", DEFAULT_RESOLUTION)
        self.declare_parameter("init_size_m", DEFAULT_INIT_SIZE)
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("update_mode", "log_odds")  # or "binary"
        self.declare_parameter("scan_stride", DEFAULT_SCAN_STRIDE)
        self.declare_parameter("l_occ", DEFAULT_L_OCC)
        self.declare_parameter("l_free", DEFAULT_L_FREE)
        self.declare_parameter("publish_rate", 2.0)  # Hz
        self.declare_parameter("max_range_m", 0.0)   # 0 -> use scan.range_max

        self.resolution: float = self.get_parameter("resolution").value
        self.init_size_m: float = self.get_parameter("init_size_m").value
        self.scan_topic: str = self.get_parameter("scan_topic").value
        self.map_topic: str = self.get_parameter("map_topic").value
        self.odom_frame: str = self.get_parameter("odom_frame").value
        self.base_frame: str = self.get_parameter("base_frame").value
        self.update_mode: str = self.get_parameter("update_mode").value
        self.scan_stride: int = max(1, int(self.get_parameter("scan_stride").value))
        self.l_occ: float = float(self.get_parameter("l_occ").value)
        self.l_free: float = float(self.get_parameter("l_free").value)
        self.publish_rate: float = float(self.get_parameter("publish_rate").value)
        self.max_range_m: float = float(self.get_parameter("max_range_m").value)

        if self.update_mode not in ("log_odds", "binary"):
            self.get_logger().warn(
                f"Unknown update_mode={self.update_mode!r} - falling back to log_odds"
            )
            self.update_mode = "log_odds"

        # ---- Grid (initialised lazily on first scan so we can centre it on
        #          the robot's current pose) -------------------------------
        self.grid: Optional[LogOddsGrid] = None
        self._scan_count = 0

        # ---- TF buffer ----------------------------------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---- ROS interfaces ---------------------------------------------
        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.scan_sub = self.create_subscription(
            LaserScan, self.scan_topic, self.on_scan, scan_qos
        )

        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.map_pub = self.create_publisher(OccupancyGrid, self.map_topic, map_qos)

        self.publish_timer = self.create_timer(
            1.0 / max(0.1, self.publish_rate), self._publish_map
        )

        self.get_logger().info(
            f"PA-4 grid mapper up.  res={self.resolution} m  "
            f"init={self.init_size_m} m  mode={self.update_mode}  "
            f"l_occ={self.l_occ:+.2f}  l_free={self.l_free:+.2f}"
        )

    # ---------------------------------------------------------------- helpers
    def _ensure_grid(self, robot_pose: Pose2D) -> None:
        if self.grid is not None:
            return
        side_cells = int(math.ceil(self.init_size_m / self.resolution))
        # Centre the initial grid on the robot.
        ox = robot_pose.x - 0.5 * side_cells * self.resolution
        oy = robot_pose.y - 0.5 * side_cells * self.resolution
        self.grid = LogOddsGrid(
            side_cells, side_cells, self.resolution, ox, oy
        )
        self.get_logger().info(
            f"Initialised {side_cells}x{side_cells} grid centred on "
            f"({robot_pose.x:.2f}, {robot_pose.y:.2f})"
        )

    def _lookup_robot_pose(self, stamp) -> Optional[Pose2D]:
        """Look up odom -> base_link.  Falls back to the latest available tf."""
        try:
            t = self.tf_buffer.lookup_transform(
                self.odom_frame, self.base_frame, stamp, timeout=rclpy.duration.Duration(seconds=0.1)
            )
        except (LookupException, ConnectivityException, ExtrapolationException):
            try:
                t = self.tf_buffer.lookup_transform(
                    self.odom_frame, self.base_frame, Time(),
                )
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                self.get_logger().warn_once(
                    f"tf {self.odom_frame}->{self.base_frame} not available: {e}"
                )
                return None
        return transform_to_pose2d(t)

    def _lookup_laser_pose(self, scan_frame: str, stamp) -> Optional[Pose2D]:
        """Look up odom -> scan_frame so we can place each ray's origin at the
        actual sensor mount, not the robot centre."""
        try:
            t = self.tf_buffer.lookup_transform(
                self.odom_frame, scan_frame, stamp, timeout=rclpy.duration.Duration(seconds=0.1)
            )
        except (LookupException, ConnectivityException, ExtrapolationException):
            try:
                t = self.tf_buffer.lookup_transform(
                    self.odom_frame, scan_frame, Time(),
                )
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                self.get_logger().warn_once(
                    f"tf {self.odom_frame}->{scan_frame} not available: {e}"
                )
                return None
        return transform_to_pose2d(t)

    # ---------------------------------------------------------------- callback
    def on_scan(self, scan: LaserScan) -> None:
        self._scan_count += 1
        if self._scan_count % self.scan_stride != 0:
            return

        laser_pose = self._lookup_laser_pose(scan.header.frame_id, scan.header.stamp)
        if laser_pose is None:
            return

        # Lazy init grid centred on the robot's first observed pose.
        if self.grid is None:
            robot_pose = self._lookup_robot_pose(scan.header.stamp) or laser_pose
            self._ensure_grid(robot_pose)
        assert self.grid is not None  # for the type checker

        max_range = self.max_range_m if self.max_range_m > 0.0 else scan.range_max
        # Clamp the practical range so a single noisy reading can't blow up
        # memory by triggering huge grid growth.
        range_max_eff = min(max_range, scan.range_max if scan.range_max > 0.0 else max_range)

        n = len(scan.ranges)
        for i, r in enumerate(scan.ranges):
            if r is None:
                continue
            # NaN / inf -> treat as max-range (ray clears free cells but no hit).
            r_clean: float
            saw_hit: bool
            if math.isnan(r) or math.isinf(r):
                r_clean = range_max_eff
                saw_hit = False
            elif r < scan.range_min or r < RANGE_MIN_FALLBACK:
                # Below sensor minimum - skip (not enough info to place a hit).
                continue
            elif r >= range_max_eff:
                r_clean = range_max_eff
                saw_hit = False
            else:
                r_clean = r
                saw_hit = True

            angle = scan.angle_min + i * scan.angle_increment
            self._integrate_ray(laser_pose, angle, r_clean, saw_hit)

    def _integrate_ray(
        self,
        laser_pose: Pose2D,
        beam_angle: float,
        beam_range: float,
        saw_hit: bool,
    ) -> None:
        """Walk one beam through the grid, updating cells along the way."""
        assert self.grid is not None

        wx0 = laser_pose.x
        wy0 = laser_pose.y
        world_angle = laser_pose.theta + beam_angle
        wx1 = wx0 + beam_range * math.cos(world_angle)
        wy1 = wy0 + beam_range * math.sin(world_angle)

        # Make sure both endpoints of the ray are inside the grid.
        c0, r0 = self.grid.world_to_grid(wx0, wy0)
        c1, r1 = self.grid.world_to_grid(wx1, wy1)
        self.grid.ensure_contains(c0, r0)
        self.grid.ensure_contains(c1, r1)
        # Re-resolve indices because growth may have shifted the origin.
        c0, r0 = self.grid.world_to_grid(wx0, wy0)
        c1, r1 = self.grid.world_to_grid(wx1, wy1)

        cells = bresenham_line(c0, r0, c1, r1)
        if not cells:
            return

        if self.update_mode == "log_odds":
            for c, r in cells[:-1]:
                self.grid.update(c, r, self.l_free)
            end_c, end_r = cells[-1]
            self.grid.update(end_c, end_r, self.l_occ if saw_hit else self.l_free)
        else:  # binary
            for c, r in cells[:-1]:
                self.grid.set_binary(c, r, occupied=False)
            end_c, end_r = cells[-1]
            if saw_hit:
                self.grid.set_binary(end_c, end_r, occupied=True)
            else:
                self.grid.set_binary(end_c, end_r, occupied=False)

    # ---------------------------------------------------------------- publish
    def _publish_map(self) -> None:
        if self.grid is None:
            return
        msg = self.grid.to_msg(self.odom_frame, self.get_clock().now().to_msg())
        self.map_pub.publish(msg)


# =============================================================================
# Main
# =============================================================================
def main(args=None) -> None:
    rclpy.init(args=args)
    node = GridMapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
