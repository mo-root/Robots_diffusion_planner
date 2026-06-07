#!/usr/bin/env python3
"""PA-3 - Path Planning and Global Navigation.

Implements the full deliberative stack required by the PA-3 rubric:

  Map pre-processing  : Gaussian-blur the raw OccupancyGrid (3x3 or 5x5),
                        republish on /map_smoothed for RViz inflation
                        visualisation.
  Planning engine     : BFS, DFS (with closed-set history), and A* using a
                        Manhattan heuristic and a heapq priority queue.
                        Algorithm choice is a ROS parameter.
  Trajectory output   : convert planned cells to a PoseArray with per-waypoint
                        yaw (each pose faces the next waypoint), publish on
                        /pose_sequence.
  Goal input          : prompt the user in the terminal for an (x, y) goal
                        in the map reference frame (no hard-coded goal).
  Robot localisation  : tf2 lookupTransform map->base_link with /odom fallback.

Extra-credit extensions (both implemented):

  8-connectivity  : parameter `connectivity` may be 4 (default) or 8; diagonal
                    moves cost sqrt(2); corner-cut prevention enforced.
  Weighted A*     : parameter `use_weighted` lets A* treat the blurred
                    occupancy as additional traversal cost, biasing the path
                    toward the centre of corridors.

Author: Moin Mattar - COSC 81/281, Spring 2026.
AI helped me with code formatting and Python style.
"""

from __future__ import annotations

import heapq
import math
import sys
import threading
from collections import deque
from typing import Iterable, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Pose, PoseArray, Quaternion, Twist
from nav_msgs.msg import OccupancyGrid, Odometry

import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


# =============================================================================
# Tuning constants (also exposed as ROS parameters where it makes sense)
# =============================================================================
# Cell values >= this are treated as occupied. Unknown (-1) is also occupied.
OCC_THRESH = 50

# Default Gaussian kernel parameters. Either 3x3 or 5x5 per the spec.
DEFAULT_KERNEL_SIZE = 5
DEFAULT_SIGMA = 1.0

# Weighted-A* multiplier for the blurred-occupancy cost: edge_cost := step *
# (1 + alpha * blurred_value). Higher alpha => stronger preference for the
# centre of corridors.
DEFAULT_WEIGHTED_ALPHA = 0.05

# Terminal-prompt timeout: seconds we'll wait on the user before bailing out.
GOAL_INPUT_TIMEOUT_S = 60.0


# =============================================================================
# Quaternion helper (yaw-only, since we drive on a flat floor)
# =============================================================================
def yaw_to_quaternion(yaw: float) -> Quaternion:
    """Return a geometry_msgs/Quaternion for a pure yaw rotation."""
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


# =============================================================================
# Gaussian kernel (no scipy/numpy dependency; pure Python)
# =============================================================================
def gaussian_kernel(size: int, sigma: float) -> List[List[float]]:
    """Return a normalised square Gaussian kernel of side `size` (must be odd)."""
    if size % 2 == 0 or size < 1:
        raise ValueError("Kernel size must be a positive odd integer")
    half = size // 2
    two_sigma2 = 2.0 * sigma * sigma
    raw = [
        [math.exp(-(r * r + c * c) / two_sigma2) for c in range(-half, half + 1)]
        for r in range(-half, half + 1)
    ]
    total = sum(v for row in raw for v in row)
    return [[v / total for v in row] for row in raw]


# =============================================================================
# Grid - occupancy grid wrapper with world<->cell conversions (cf. lec10-map.py)
# =============================================================================
class Grid:
    """Row-major occupancy grid with float values 0..100.

    Stored values are floats so the same class works for the raw integer
    OccupancyGrid (0/100/-1) and for the Gaussian-blurred output.
    """

    def __init__(
        self,
        data: Iterable[float],
        width: int,
        height: int,
        resolution: float,
        origin_x: float,
        origin_y: float,
    ) -> None:
        self.data: List[float] = [float(v) for v in data]
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y

    # --- world <-> cell conversions ---
    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        """Return the (col, row) cell containing world point (wx, wy)."""
        c = int(math.floor((wx - self.origin_x) / self.resolution))
        r = int(math.floor((wy - self.origin_y) / self.resolution))
        return c, r

    def grid_to_world(self, c: int, r: int) -> Tuple[float, float]:
        """Return the world (x, y) at the centre of cell (c, r)."""
        wx = self.origin_x + (c + 0.5) * self.resolution
        wy = self.origin_y + (r + 0.5) * self.resolution
        return wx, wy

    # --- bounds + occupancy queries ---
    def in_bounds(self, c: int, r: int) -> bool:
        return 0 <= c < self.width and 0 <= r < self.height

    def cell_value(self, c: int, r: int) -> float:
        return self.data[r * self.width + c]

    def is_free_cell(self, c: int, r: int) -> bool:
        """True iff (c, r) is in bounds and below the occupancy threshold.

        Unknown cells (negative values, e.g. -1) are treated as occupied.
        """
        if not self.in_bounds(c, r):
            return False
        v = self.cell_value(c, r)
        return 0.0 <= v < OCC_THRESH

    # --- Gaussian convolution ---
    def gaussian_blurred(self, kernel_size: int, sigma: float) -> "Grid":
        """Return a new Grid with the same metadata, blurred with a Gaussian.

        The input is binarised first (occupied/unknown -> 100, free -> 0) so
        that the blurred output is an "inflated" map: cells near obstacles
        have positive values that fade with distance, exactly the safety
        buffer the spec asks for.
        """
        k = gaussian_kernel(kernel_size, sigma)
        half = kernel_size // 2
        w, h = self.width, self.height
        binary = [
            100.0 if (v < 0.0 or v >= OCC_THRESH) else 0.0 for v in self.data
        ]
        out: List[float] = [0.0] * (w * h)
        for r in range(h):
            for c in range(w):
                acc = 0.0
                for dr in range(-half, half + 1):
                    for dc in range(-half, half + 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            acc += k[dr + half][dc + half] * binary[nr * w + nc]
                        else:
                            # Treat out-of-bounds as obstacle so the border is
                            # blurred properly inwards.
                            acc += k[dr + half][dc + half] * 100.0
                out[r * w + c] = acc
        return Grid(out, w, h, self.resolution, self.origin_x, self.origin_y)


# =============================================================================
# Neighbour generator
# =============================================================================
SQRT2 = math.sqrt(2.0)


def neighbours(c: int, r: int, connectivity: int) -> List[Tuple[Tuple[int, int], float]]:
    """Return [(neighbour_cell, step_cost), ...] for 4- or 8-connectivity."""
    if connectivity == 4:
        return [
            ((c + 1, r), 1.0), ((c - 1, r), 1.0),
            ((c, r + 1), 1.0), ((c, r - 1), 1.0),
        ]
    if connectivity == 8:
        return [
            ((c + 1, r), 1.0), ((c - 1, r), 1.0),
            ((c, r + 1), 1.0), ((c, r - 1), 1.0),
            ((c + 1, r + 1), SQRT2), ((c + 1, r - 1), SQRT2),
            ((c - 1, r + 1), SQRT2), ((c - 1, r - 1), SQRT2),
        ]
    raise ValueError("connectivity must be 4 or 8")


# =============================================================================
# Search algorithms - all return (path_cells, num_expansions)
# =============================================================================
def _reconstruct(came_from: dict, cur: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Walk the parent dict back to the start, then reverse."""
    cells = [cur]
    while came_from.get(cur) is not None:
        cur = came_from[cur]
        cells.append(cur)
    cells.reverse()
    return cells


def bfs(
    grid: Grid,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    connectivity: int = 4,
) -> Tuple[Optional[List[Tuple[int, int]]], int]:
    """Breadth-first search with closed-set history.

    Returns (path_cells_from_start_to_goal, num_expansions) or (None, n) if
    no path exists. BFS finds the path with fewest cells (optimal under unit
    step cost).
    """
    if not grid.is_free_cell(*start) or not grid.is_free_cell(*goal):
        return None, 0
    closed: set = set()
    queue: deque = deque([start])
    came_from: dict = {start: None}
    expansions = 0
    while queue:
        cur = queue.popleft()
        if cur in closed:
            continue
        closed.add(cur)
        expansions += 1
        if cur == goal:
            return _reconstruct(came_from, cur), expansions
        for nb, _cost in neighbours(cur[0], cur[1], connectivity):
            if nb in closed or nb in came_from:
                continue
            if not grid.is_free_cell(*nb):
                continue
            came_from[nb] = cur
            queue.append(nb)
    return None, expansions


def dfs(
    grid: Grid,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    connectivity: int = 4,
) -> Tuple[Optional[List[Tuple[int, int]]], int]:
    """Depth-first search WITH closed-set history.

    The closed set is the algorithmic difference from textbook tree-search DFS
    that the HW4 Q2 analysis warns about: without it, DFS on a finite cyclic
    graph can loop forever.
    """
    if not grid.is_free_cell(*start) or not grid.is_free_cell(*goal):
        return None, 0
    closed: set = set()
    stack: List[Tuple[int, int]] = [start]
    came_from: dict = {start: None}
    expansions = 0
    while stack:
        cur = stack.pop()
        if cur in closed:
            continue
        closed.add(cur)
        expansions += 1
        if cur == goal:
            return _reconstruct(came_from, cur), expansions
        for nb, _cost in neighbours(cur[0], cur[1], connectivity):
            if nb in closed:
                continue
            if not grid.is_free_cell(*nb):
                continue
            if nb not in came_from:
                came_from[nb] = cur
                stack.append(nb)
    return None, expansions


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(
    grid: Grid,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    connectivity: int = 4,
    blurred: Optional[Grid] = None,
    weighted_alpha: float = 0.0,
) -> Tuple[Optional[List[Tuple[int, int]]], int]:
    """A* with a Manhattan heuristic and a binary-heap priority queue.

    `connectivity`     : 4 (default) or 8.
    `blurred` + alpha  : weighted-A* extra credit. When `blurred` is provided
                         the per-edge cost is multiplied by
                         (1 + alpha * blurred[neighbour]), so cells with
                         high blurred-occupancy (i.e. close to walls) are
                         penalised and the planner prefers corridor centres.

    The priority-queue entry is (f, h, counter, g, cell) so ties on f are
    broken first by h (smaller h wins) and then by FIFO insertion order
    (the integer counter), giving stable, reproducible orderings.
    """
    if not grid.is_free_cell(*start) or not grid.is_free_cell(*goal):
        return None, 0
    open_heap: List[Tuple[float, float, int, float, Tuple[int, int]]] = []
    counter = 0
    h0 = manhattan(start, goal)
    heapq.heappush(open_heap, (h0, h0, counter, 0.0, start))
    g_score = {start: 0.0}
    came_from: dict = {start: None}
    closed: set = set()
    expansions = 0
    while open_heap:
        f, _h, _c, g, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        closed.add(cur)
        expansions += 1
        if cur == goal:
            return _reconstruct(came_from, cur), expansions
        for nb, step in neighbours(cur[0], cur[1], connectivity):
            if not grid.is_free_cell(*nb):
                continue
            # Corner-cut prevention: a diagonal move is illegal if either
            # axis-aligned neighbour through which it "slips" is occupied.
            if connectivity == 8 and abs(nb[0] - cur[0]) == 1 and abs(nb[1] - cur[1]) == 1:
                if not grid.is_free_cell(nb[0], cur[1]):
                    continue
                if not grid.is_free_cell(cur[0], nb[1]):
                    continue
            cost = step
            if blurred is not None:
                cost *= 1.0 + weighted_alpha * blurred.cell_value(*nb)
            tentative = g + cost
            if tentative < g_score.get(nb, math.inf):
                g_score[nb] = tentative
                came_from[nb] = cur
                counter += 1
                h_nb = manhattan(nb, goal)
                heapq.heappush(open_heap, (tentative + h_nb, h_nb, counter, tentative, nb))
    return None, expansions


# =============================================================================
# ROS 2 node
# =============================================================================
class PathPlanner(Node):
    """Subscribes to /map and /odom; plans once with the chosen algorithm and
    publishes /map_smoothed and /pose_sequence."""

    def __init__(self) -> None:
        super().__init__("pa3_path_planner")

        # --- Parameters (all overridable from the command line) ----------
        self.declare_parameter("algorithm", "astar")        # 'bfs' | 'dfs' | 'astar'
        self.declare_parameter("connectivity", 4)            # 4 or 8
        self.declare_parameter("kernel_size", DEFAULT_KERNEL_SIZE)
        self.declare_parameter("sigma", DEFAULT_SIGMA)
        self.declare_parameter("use_weighted", False)
        self.declare_parameter("weighted_alpha", DEFAULT_WEIGHTED_ALPHA)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        # Follower (drives the robot through the planned waypoints so the
        # video deliverable shows actual motion, not just a published PoseArray).
        self.declare_parameter("drive_robot", True)
        self.declare_parameter("follow_speed", 0.4)         # m/s
        self.declare_parameter("follow_kp_heading", 1.6)    # rad/s per rad
        self.declare_parameter("follow_goal_tolerance", 0.25)  # m
        self.declare_parameter("follow_lookahead", 0.5)     # m
        # If non-NaN, skip the terminal prompt (useful for scripted testing).
        self.declare_parameter("goal_x", float("nan"))
        self.declare_parameter("goal_y", float("nan"))
        # Optional explicit start pose in the *map* frame (used when tf2 isn't
        # available, e.g. running with Stage's prefixed frames without a
        # localisation node). NaN means "use tf2 / odom".
        self.declare_parameter("start_x", float("nan"))
        self.declare_parameter("start_y", float("nan"))

        # --- ROS interfaces ---------------------------------------------
        # /map is published transient_local + reliable by nav2_map_server.
        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )

        self._map_sub = self.create_subscription(
            OccupancyGrid, "/map", self._on_map, latched_qos
        )
        self._odom_sub = self.create_subscription(
            Odometry, "/odom", self._on_odom, 10
        )
        self._smoothed_pub = self.create_publisher(
            OccupancyGrid, "/map_smoothed", latched_qos
        )
        self._pose_seq_pub = self.create_publisher(
            PoseArray, "/pose_sequence", latched_qos
        )
        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # tf2 buffer for map->base_link lookup.
        self._tf_buf = tf2_ros.Buffer()
        self._tf_lis = tf2_ros.TransformListener(self._tf_buf, self)

        # --- State ------------------------------------------------------
        self._raw_grid: Optional[Grid] = None
        self._smoothed_grid: Optional[Grid] = None
        self._odom_pose: Optional[Tuple[float, float, float]] = None  # (x, y, yaw)
        self._goal_world: Optional[Tuple[float, float]] = None
        self._planned = False
        # Follower state: list of waypoints (world m) + current target index.
        self._waypoints: Optional[List[Tuple[float, float]]] = None
        self._wp_idx = 0
        self._follow_timer = self.create_timer(0.1, self._follow_tick)

        # The terminal prompt blocks, so run it on a worker thread.
        self._goal_thread = threading.Thread(target=self._prompt_goal, daemon=True)
        self._goal_thread.start()

        self.get_logger().info(
            "pa3_path_planner started. Waiting for /map and goal input..."
        )

    # =====================================================================
    # Goal acquisition (terminal prompt)
    # =====================================================================
    def _prompt_goal(self) -> None:
        """Read goal (x, y) in the map frame from stdin.

        If the `goal_x` / `goal_y` parameters are set, use them and skip
        the prompt entirely (handy for non-interactive testing).
        """
        gx_p = float(self.get_parameter("goal_x").value)
        gy_p = float(self.get_parameter("goal_y").value)
        if not (math.isnan(gx_p) or math.isnan(gy_p)):
            self._goal_world = (gx_p, gy_p)
            self.get_logger().info(
                f"goal taken from parameters: ({gx_p:.2f}, {gy_p:.2f}) m"
            )
            self._try_plan()
            return

        try:
            print("\n=== PA-3 goal entry =========================================")
            print("Enter goal coordinates in the *map* frame (metres).")
            gx = float(input("  goal_x: ").strip())
            gy = float(input("  goal_y: ").strip())
        except (EOFError, ValueError) as exc:
            self.get_logger().error(f"goal input failed: {exc}")
            return
        self._goal_world = (gx, gy)
        self.get_logger().info(f"goal entered: ({gx:.2f}, {gy:.2f}) m")
        self._try_plan()

    # =====================================================================
    # Map / odom callbacks
    # =====================================================================
    def _on_map(self, msg: OccupancyGrid) -> None:
        if self._raw_grid is not None:
            return  # plan once - the rubric assumes a static map
        self.get_logger().info(
            f"map received: {msg.info.width}x{msg.info.height} @ "
            f"{msg.info.resolution:.3f} m/cell, origin=("
            f"{msg.info.origin.position.x:.2f}, {msg.info.origin.position.y:.2f})"
        )
        self._raw_grid = Grid(
            msg.data,
            msg.info.width, msg.info.height,
            msg.info.resolution,
            msg.info.origin.position.x, msg.info.origin.position.y,
        )
        # Run Gaussian blur and republish.
        ksize = int(self.get_parameter("kernel_size").value)
        sigma = float(self.get_parameter("sigma").value)
        self._smoothed_grid = self._raw_grid.gaussian_blurred(ksize, sigma)
        self._publish_smoothed_map(msg.header.frame_id)
        self._try_plan()

    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        # quaternion -> yaw (only z-rotation matters on a flat floor)
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        self._odom_pose = (p.x, p.y, yaw)
        # If the map and goal are already known but planning waited on a
        # robot pose, kick it off now.
        if (
            self._raw_grid is not None
            and self._goal_world is not None
            and not self._planned
        ):
            self._try_plan()

    # =====================================================================
    # /map_smoothed publication
    # =====================================================================
    def _publish_smoothed_map(self, frame_id: str) -> None:
        assert self._smoothed_grid is not None
        msg = OccupancyGrid()
        msg.header.frame_id = frame_id or "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = self._smoothed_grid.resolution
        msg.info.width = self._smoothed_grid.width
        msg.info.height = self._smoothed_grid.height
        msg.info.origin.position.x = self._smoothed_grid.origin_x
        msg.info.origin.position.y = self._smoothed_grid.origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        # OccupancyGrid is int8 [-1, 100]; clamp the float result.
        msg.data = [
            max(0, min(100, int(round(v))))
            for v in self._smoothed_grid.data
        ]
        self._smoothed_pub.publish(msg)
        self.get_logger().info(
            f"published /map_smoothed ({len(msg.data)} cells, "
            f"max value {max(msg.data)})"
        )

    # =====================================================================
    # Robot localisation: tf2 first, /odom fallback
    # =====================================================================
    def _robot_position_in_map(self) -> Optional[Tuple[float, float]]:
        """Return (x, y) of base_link in the map frame, or None if unavailable."""
        # Explicit override (parameters), if provided.
        sx = float(self.get_parameter("start_x").value)
        sy = float(self.get_parameter("start_y").value)
        if not (math.isnan(sx) or math.isnan(sy)):
            return (sx, sy)
        map_frame = str(self.get_parameter("map_frame").value)
        base_frame = str(self.get_parameter("base_frame").value)
        try:
            t = self._tf_buf.lookup_transform(
                map_frame, base_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2),
            )
            return (t.transform.translation.x, t.transform.translation.y)
        except (LookupException, ConnectivityException, ExtrapolationException) as exc:
            self.get_logger().debug(f"tf lookup failed ({exc}); falling back to /odom")
            if self._odom_pose is not None:
                return (self._odom_pose[0], self._odom_pose[1])
            return None

    # =====================================================================
    # Planning
    # =====================================================================
    def _try_plan(self) -> None:
        if self._planned:
            return
        if self._raw_grid is None or self._smoothed_grid is None:
            return
        if self._goal_world is None:
            return
        start_world = self._robot_position_in_map()
        if start_world is None:
            self.get_logger().info("waiting for tf or /odom to know the start pose...")
            return

        algo = str(self.get_parameter("algorithm").value).lower()
        connectivity = int(self.get_parameter("connectivity").value)
        if connectivity not in (4, 8):
            self.get_logger().warn(
                f"connectivity={connectivity} is invalid; falling back to 4"
            )
            connectivity = 4
        use_weighted = bool(self.get_parameter("use_weighted").value)
        weighted_alpha = float(self.get_parameter("weighted_alpha").value)

        # Plan in the SMOOTHED map (so safety inflation is automatic).
        plan_grid = self._smoothed_grid
        start_cell = plan_grid.world_to_grid(*start_world)
        goal_cell = plan_grid.world_to_grid(*self._goal_world)

        self.get_logger().info(
            f"planning with algorithm='{algo}', connectivity={connectivity}, "
            f"weighted={use_weighted} | "
            f"start_world=({start_world[0]:.2f},{start_world[1]:.2f})"
            f" -> cell {start_cell}, "
            f"goal_world=({self._goal_world[0]:.2f},{self._goal_world[1]:.2f})"
            f" -> cell {goal_cell}"
        )

        if algo == "bfs":
            cells, expansions = bfs(plan_grid, start_cell, goal_cell, connectivity)
        elif algo == "dfs":
            cells, expansions = dfs(plan_grid, start_cell, goal_cell, connectivity)
        elif algo == "astar":
            blurred = self._smoothed_grid if use_weighted else None
            cells, expansions = astar(
                plan_grid, start_cell, goal_cell, connectivity,
                blurred=blurred, weighted_alpha=weighted_alpha,
            )
        else:
            self.get_logger().error(
                f"unknown algorithm '{algo}'; expected 'bfs', 'dfs', or 'astar'"
            )
            return

        if cells is None:
            self.get_logger().error(
                f"{algo.upper()} found NO path after {expansions} expansions"
            )
            self._planned = True  # don't keep retrying
            return

        self.get_logger().info(
            f"{algo.upper()} found path: {len(cells)} cells, "
            f"{expansions} expansions"
        )
        self._publish_pose_sequence(cells, plan_grid)
        # Stash the world-frame waypoints for the follower (drives the robot
        # through them so the video shows real motion).
        self._waypoints = [plan_grid.grid_to_world(c, r) for (c, r) in cells]
        self._wp_idx = 0
        self._planned = True

    # =====================================================================
    # /pose_sequence publication
    # =====================================================================
    def _publish_pose_sequence(
        self, cells: List[Tuple[int, int]], grid: Grid
    ) -> None:
        """Convert cells to a PoseArray; each pose's yaw faces the next waypoint.

        The last pose inherits the previous pose's yaw (no "next waypoint" to
        face), keeping orientation continuous.
        """
        msg = PoseArray()
        msg.header.frame_id = self.get_parameter("map_frame").value
        msg.header.stamp = self.get_clock().now().to_msg()

        worlds = [grid.grid_to_world(c, r) for (c, r) in cells]
        prev_yaw = 0.0
        for i, (wx, wy) in enumerate(worlds):
            if i + 1 < len(worlds):
                nx, ny = worlds[i + 1]
                yaw = math.atan2(ny - wy, nx - wx)
                prev_yaw = yaw
            else:
                yaw = prev_yaw  # last waypoint: inherit
            p = Pose()
            p.position.x = wx
            p.position.y = wy
            p.position.z = 0.0
            p.orientation = yaw_to_quaternion(yaw)
            msg.poses.append(p)
        self._pose_seq_pub.publish(msg)
        self.get_logger().info(
            f"published /pose_sequence with {len(msg.poses)} waypoints"
        )

    # =====================================================================
    # Pure-pursuit follower (drives the robot through the planned waypoints)
    # =====================================================================
    def _follow_tick(self) -> None:
        """10 Hz: walk toward the current waypoint; advance when close enough.

        Skipped entirely if `drive_robot` is False (planner-only mode).
        """
        if not bool(self.get_parameter("drive_robot").value):
            return
        if self._waypoints is None or self._odom_pose is None:
            return
        if self._wp_idx >= len(self._waypoints):
            self._cmd_pub.publish(Twist())  # stop
            return

        speed = float(self.get_parameter("follow_speed").value)
        kp_h = float(self.get_parameter("follow_kp_heading").value)
        tol = float(self.get_parameter("follow_goal_tolerance").value)
        lookahead = float(self.get_parameter("follow_lookahead").value)

        # Convert odom-frame pose to map-frame using the explicit start_x/y
        # offset (since Stage's odom origin is at the robot's spawn pose in
        # the map, not at the map origin).
        sx = float(self.get_parameter("start_x").value)
        sy = float(self.get_parameter("start_y").value)
        offx = sx if not math.isnan(sx) else 0.0
        offy = sy if not math.isnan(sy) else 0.0
        rx = self._odom_pose[0] + offx
        ry = self._odom_pose[1] + offy
        ryaw = self._odom_pose[2]
        # Skip-ahead: if the next waypoint is already inside the lookahead
        # radius, advance to the one after.
        while self._wp_idx < len(self._waypoints) - 1:
            wx, wy = self._waypoints[self._wp_idx]
            if math.hypot(wx - rx, wy - ry) < lookahead:
                self._wp_idx += 1
            else:
                break
        wx, wy = self._waypoints[self._wp_idx]
        dx, dy = wx - rx, wy - ry
        dist = math.hypot(dx, dy)

        # Final goal reached?
        if self._wp_idx == len(self._waypoints) - 1 and dist < tol:
            self.get_logger().info("goal reached; stopping")
            self._cmd_pub.publish(Twist())
            self._wp_idx = len(self._waypoints)  # mark done
            return

        target_yaw = math.atan2(dy, dx)
        heading_err = (target_yaw - ryaw + math.pi) % (2 * math.pi) - math.pi
        cmd = Twist()
        cmd.angular.z = max(-1.5, min(1.5, kp_h * heading_err))
        # Slow down when heading is far off so we turn-on-the-spot first.
        cmd.linear.x = speed * max(0.0, math.cos(heading_err))
        self._cmd_pub.publish(cmd)


# =============================================================================
# main
# =============================================================================
def main(args=None) -> None:
    rclpy.init(args=args)
    node = PathPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
