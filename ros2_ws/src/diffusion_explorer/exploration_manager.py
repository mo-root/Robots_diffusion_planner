#!/usr/bin/env python3
"""ROS 2 exploration manager: plans an A* path on the live /map to each
diffusion-scored frontier and follows it with pure-pursuit.

This is the planning + control half of the integrated system. It closes the
loop with the mapping module: every goal is reached by running A* graph search
(the same algorithm as PA3) over the **real-time occupancy grid** published by
the PA4 mapper, so the planner actively consumes mapping updates and the robot
routes around walls rather than driving blindly toward the goal.

Subscribes to:
  /best_frontier (geometry_msgs/PoseStamped) — goal, from diffusion_frontier_node
  /map           (nav_msgs/OccupancyGrid)    — live grid from the PA4 mapper
  /odom          (nav_msgs/Odometry)         — robot pose

Publishes:
  /cmd_vel       (geometry_msgs/Twist) — velocity commands (pure-pursuit)
  /planned_path  (nav_msgs/Path)       — the A* path (for RViz)
  /exploration_done (std_msgs/Bool)
"""

import heapq
import math
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from std_msgs.msg import Bool
import tf2_ros


class ExplorationManager(Node):
    def __init__(self):
        super().__init__("exploration_manager")

        self.declare_parameter("max_linear_speed", 0.2)
        self.declare_parameter("max_angular_speed", 0.5)
        self.declare_parameter("goal_tolerance", 0.5)
        self.declare_parameter("waypoint_tolerance", 0.25)
        self.declare_parameter("coverage_threshold", 0.90)
        self.declare_parameter("occ_threshold", 50)      # >= this (0-100) = obstacle
        self.declare_parameter("inflate_radius", 4)       # obstacle inflation, cells (>= robot radius)
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("stuck_timeout", 6.0)      # s without progress -> recover
        self.declare_parameter("recovery_time", 1.6)      # s of reverse + rotate
        self.declare_parameter("progress_dist", 0.15)     # m that counts as progress

        self.max_v = self.get_parameter("max_linear_speed").value
        self.max_w = self.get_parameter("max_angular_speed").value
        self.goal_tol = self.get_parameter("goal_tolerance").value
        self.wp_tol = self.get_parameter("waypoint_tolerance").value
        self.coverage_thresh = self.get_parameter("coverage_threshold").value
        self.occ_thresh = self.get_parameter("occ_threshold").value
        self.inflate = self.get_parameter("inflate_radius").value
        self.odom_frame = self.get_parameter("odom_frame").value
        self.stuck_timeout = self.get_parameter("stuck_timeout").value
        self.recovery_time = self.get_parameter("recovery_time").value
        self.progress_dist = self.get_parameter("progress_dist").value

        self.goal_sub = self.create_subscription(
            PoseStamped, "/best_frontier", self.goal_callback, 10)
        self.map_sub = self.create_subscription(
            OccupancyGrid, "/map", self.map_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 10)

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.path_pub = self.create_publisher(Path, "/planned_path", 10)
        self.done_pub = self.create_publisher(Bool, "/exploration_done", 10)

        self.control_timer = self.create_timer(0.1, self.control_loop)
        self.status_timer = self.create_timer(10.0, self.status_update)

        self.current_goal = None
        self.robot_x = self.robot_y = self.robot_yaw = 0.0
        self.coverage = 0.0
        self.exploration_done = False
        self.goals_reached = 0

        # live map (set in map_callback)
        self.grid = None              # 2D int8 (H, W); -1 unknown, 0..100 occ prob
        self.map_res = None
        self.map_ox = self.map_oy = 0.0
        # active A* plan
        self.path = []                # list of (wx, wy) world waypoints
        self.path_idx = 0
        self.need_plan = False
        # stuck detection / recovery
        self.last_prog_pos = (0.0, 0.0)
        self.last_prog_time = 0.0
        self.recovery_until = 0.0
        self.blacklist = []           # (wx, wy, expiry) goals abandoned as unreachable

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.get_logger().info("ExplorationManager ready (A* planning on /map)")

    # ---------------- callbacks ----------------
    def goal_callback(self, msg: PoseStamped):
        # Commit to the current goal until it is reached or abandoned; ignore the
        # scorer's intermediate updates so the robot does not oscillate between
        # frontiers and actually finishes exploring each one.
        if self.exploration_done or self.current_goal is not None:
            return
        gx, gy = msg.pose.position.x, msg.pose.position.y
        now = self.get_clock().now().nanoseconds * 1e-9
        self.blacklist = [(bx, by, e) for (bx, by, e) in self.blacklist if e > now]
        for (bx, by, _e) in self.blacklist:
            if math.hypot(gx - bx, gy - by) < 0.6:
                return                # ignore a frontier we just abandoned as unreachable
        self.current_goal = (gx, gy)
        self.need_plan = True         # replan on every new scored goal
        self.get_logger().info(f"New goal: ({gx:.2f}, {gy:.2f})")

    def odom_callback(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny, cosy)

    def map_callback(self, msg: OccupancyGrid):
        w, h = msg.info.width, msg.info.height
        self.grid = np.array(msg.data, dtype=np.int8).reshape(h, w)
        self.map_res = msg.info.resolution
        self.map_ox = msg.info.origin.position.x
        self.map_oy = msg.info.origin.position.y
        known = int(np.sum(self.grid != -1))
        self.coverage = known / self.grid.size if self.grid.size else 0.0

    # ---------------- grid <-> world ----------------
    def _w2g(self, wx, wy):
        return (int((wy - self.map_oy) / self.map_res),
                int((wx - self.map_ox) / self.map_res))    # (row=gy, col=gx)

    def _g2w(self, gy, gx):
        return (self.map_ox + (gx + 0.5) * self.map_res,
                self.map_oy + (gy + 0.5) * self.map_res)

    def _traversable(self):
        """Boolean free-space grid with obstacles inflated by robot radius."""
        occ = self.grid >= self.occ_thresh                 # known obstacle
        for _ in range(max(0, self.inflate)):              # cheap 4-connected dilation
            occ[1:, :] |= occ[:-1, :].copy()
            occ[:-1, :] |= occ[1:, :].copy()
            occ[:, 1:] |= occ[:, :-1].copy()
            occ[:, :-1] |= occ[:, 1:].copy()
        free = (self.grid >= 0) & (self.grid < self.occ_thresh)  # known free
        return free & ~occ

    # ---------------- A* (8-connected, no corner cutting) ----------------
    def _snap(self, trav, cell):
        gy, gx = cell
        H, W = trav.shape
        if 0 <= gy < H and 0 <= gx < W and trav[gy, gx]:
            return (gy, gx)
        ys, xs = np.where(trav)
        if len(ys) == 0:
            return None
        i = ((ys - gy) ** 2 + (xs - gx) ** 2).argmin()
        return (int(ys[i]), int(xs[i]))

    def _astar(self, trav, start, goal):
        H, W = trav.shape
        if start is None or goal is None or not (trav[start] and trav[goal]):
            return None
        nbrs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]
        openq, g, came = [(0.0, start)], {start: 0.0}, {}
        while openq:
            _, cur = heapq.heappop(openq)
            if cur == goal:
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                return path[::-1]
            cy, cx = cur
            for dy, dx, c in nbrs:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < H and 0 <= nx < W and trav[ny, nx]:
                    if dy and dx and not (trav[cy + dy, cx] and trav[cy, cx + dx]):
                        continue
                    ng = g[cur] + c
                    if ng < g.get((ny, nx), 1e18):
                        g[(ny, nx)] = ng
                        came[(ny, nx)] = cur
                        heapq.heappush(openq, (ng + math.hypot(ny - goal[0], nx - goal[1]), (ny, nx)))
        return None

    def plan_path(self):
        """Run A* on the current /map to the goal; store world waypoints + publish."""
        self.need_plan = False
        self.path, self.path_idx = [], 0
        if self.grid is None or self.current_goal is None:
            return False
        trav = self._traversable()
        start = self._snap(trav, self._w2g(self.robot_x, self.robot_y))
        goal = self._snap(trav, self._w2g(*self.current_goal))
        cells = self._astar(trav, start, goal)
        if not cells or len(cells) < 2:
            self.get_logger().warn("A* found no path on /map; will retry on next map update")
            return False
        # downsample to ~1 waypoint every 6 cells, always keep the last
        wps = [self._g2w(gy, gx) for (gy, gx) in cells[::6]]
        wps.append(self._g2w(*cells[-1]))
        self.path, self.path_idx = wps, 0
        self._publish_path(wps)
        self.get_logger().info(f"Planned A* path: {len(cells)} cells -> {len(wps)} waypoints")
        return True

    # ---------------- control ----------------
    def control_loop(self):
        if self.exploration_done:
            return
        if self.coverage >= self.coverage_thresh:
            self.get_logger().info(f"Coverage {self.coverage:.0%} reached. Done!")
            self.exploration_done = True
            self._stop()
            self.done_pub.publish(Bool(data=True))
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        # recovery: reverse + rotate to escape a wall the robot has wedged against
        if now < self.recovery_until:
            cmd = Twist()
            cmd.linear.x = -0.12
            cmd.angular.z = 0.9
            self.cmd_pub.publish(cmd)
            return
        # stuck detection: no real progress for stuck_timeout seconds
        if math.hypot(self.robot_x - self.last_prog_pos[0],
                      self.robot_y - self.last_prog_pos[1]) > self.progress_dist:
            self.last_prog_pos = (self.robot_x, self.robot_y)
            self.last_prog_time = now
        elif self.current_goal is not None and now - self.last_prog_time > self.stuck_timeout:
            self.get_logger().warn("No progress; recovering and abandoning current goal.")
            self.blacklist.append((self.current_goal[0], self.current_goal[1], now + 25.0))
            self.recovery_until = now + self.recovery_time
            self.current_goal, self.path = None, []
            self.last_prog_time = now
            self.last_prog_pos = (self.robot_x, self.robot_y)
            return

        if self.current_goal is None:
            return

        # (re)plan when a new goal arrived or we have no active path
        if self.need_plan or not self.path:
            if not self.plan_path():
                return

        # reached the goal?
        if math.hypot(self.current_goal[0] - self.robot_x,
                      self.current_goal[1] - self.robot_y) < self.goal_tol:
            self.goals_reached += 1
            self.get_logger().info(
                f"Reached goal #{self.goals_reached} | coverage: {self.coverage:.0%}")
            self.current_goal, self.path = None, []
            self._stop()
            return

        # advance through waypoints we've already passed
        while self.path_idx < len(self.path) - 1:
            wx, wy = self.path[self.path_idx]
            if math.hypot(wx - self.robot_x, wy - self.robot_y) < self.wp_tol:
                self.path_idx += 1
            else:
                break

        tx, ty = self.path[self.path_idx]
        self._drive_toward(tx, ty)

    def _drive_toward(self, tx, ty):
        dx, dy = tx - self.robot_x, ty - self.robot_y
        dist = math.hypot(dx, dy)
        angle_err = math.atan2(dy, dx) - self.robot_yaw
        angle_err = (angle_err + math.pi) % (2 * math.pi) - math.pi
        cmd = Twist()
        if abs(angle_err) > 0.4:
            cmd.angular.z = max(-self.max_w, min(self.max_w, angle_err))
            cmd.linear.x = 0.05
        else:
            cmd.linear.x = min(self.max_v, dist * 0.6)
            cmd.angular.z = max(-self.max_w, min(self.max_w, angle_err * 2.0))
        self.cmd_pub.publish(cmd)

    def _stop(self):
        self.cmd_pub.publish(Twist())

    def _publish_path(self, wps):
        path = Path()
        path.header.frame_id = self.odom_frame
        path.header.stamp = self.get_clock().now().to_msg()
        for (wx, wy) in wps:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x, ps.pose.position.y = wx, wy
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.path_pub.publish(path)

    def status_update(self):
        goal_str = (f"({self.current_goal[0]:.1f}, {self.current_goal[1]:.1f})"
                    if self.current_goal else "none")
        self.get_logger().info(
            f"Coverage: {self.coverage:.0%} | goals reached: {self.goals_reached} | "
            f"goal: {goal_str} | waypoints left: {max(0, len(self.path) - self.path_idx)}")


def main(args=None):
    rclpy.init(args=args)
    node = ExplorationManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
