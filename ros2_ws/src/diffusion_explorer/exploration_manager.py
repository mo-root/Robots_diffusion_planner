#!/usr/bin/env python3
"""ROS 2 exploration manager: drives the robot to diffusion-scored frontiers.

Subscribes to:
  /best_frontier (geometry_msgs/PoseStamped) — from diffusion_frontier_node
  /map (nav_msgs/OccupancyGrid) — for coverage tracking

Publishes:
  /cmd_vel (geometry_msgs/Twist) — velocity commands (pure pursuit)

This is a simplified pure-pursuit controller that drives toward the
best frontier goal. In the full system, this would use PA3's A* planner.
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import Bool
import tf2_ros


class ExplorationManager(Node):
    def __init__(self):
        super().__init__("exploration_manager")

        self.declare_parameter("max_linear_speed", 0.2)
        self.declare_parameter("max_angular_speed", 0.5)
        self.declare_parameter("goal_tolerance", 0.5)
        self.declare_parameter("lookahead", 0.8)
        self.declare_parameter("coverage_threshold", 0.90)
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")

        self.max_v = self.get_parameter("max_linear_speed").value
        self.max_w = self.get_parameter("max_angular_speed").value
        self.goal_tol = self.get_parameter("goal_tolerance").value
        self.coverage_thresh = self.get_parameter("coverage_threshold").value

        self.goal_sub = self.create_subscription(
            PoseStamped, "/best_frontier", self.goal_callback, 10)
        self.map_sub = self.create_subscription(
            OccupancyGrid, "/map", self.map_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 10)

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.done_pub = self.create_publisher(Bool, "/exploration_done", 10)

        self.control_timer = self.create_timer(0.1, self.control_loop)
        self.status_timer = self.create_timer(10.0, self.status_update)

        self.current_goal = None
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.coverage = 0.0
        self.exploration_done = False
        self.goals_reached = 0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info("ExplorationManager ready")

    def goal_callback(self, msg: PoseStamped):
        if self.exploration_done:
            return
        self.current_goal = (msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(
            f"New goal: ({self.current_goal[0]:.2f}, {self.current_goal[1]:.2f})")

    def odom_callback(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny, cosy)

    def map_callback(self, msg: OccupancyGrid):
        data = np.array(msg.data, dtype=np.int8)
        known = np.sum(data != -1)
        total = len(data)
        self.coverage = known / total if total > 0 else 0.0

    def control_loop(self):
        if self.exploration_done:
            return

        if self.coverage >= self.coverage_thresh:
            self.get_logger().info(
                f"Coverage {self.coverage:.0%} >= {self.coverage_thresh:.0%}. Done!")
            self.exploration_done = True
            self._stop()
            done_msg = Bool()
            done_msg.data = True
            self.done_pub.publish(done_msg)
            return

        if self.current_goal is None:
            return

        gx, gy = self.current_goal
        dx = gx - self.robot_x
        dy = gy - self.robot_y
        dist = math.sqrt(dx*dx + dy*dy)

        if dist < self.goal_tol:
            self.goals_reached += 1
            self.get_logger().info(
                f"Reached goal #{self.goals_reached} | coverage: {self.coverage:.0%}")
            self.current_goal = None
            self._stop()
            return

        angle_to_goal = math.atan2(dy, dx)
        angle_error = angle_to_goal - self.robot_yaw
        while angle_error > math.pi:
            angle_error -= 2 * math.pi
        while angle_error < -math.pi:
            angle_error += 2 * math.pi

        cmd = Twist()

        if abs(angle_error) > 0.4:
            cmd.angular.z = max(-self.max_w, min(self.max_w, angle_error))
            cmd.linear.x = 0.05
        else:
            cmd.linear.x = min(self.max_v, dist * 0.5)
            cmd.angular.z = max(-self.max_w, min(self.max_w, angle_error * 2.0))

        self.cmd_pub.publish(cmd)

    def _stop(self):
        cmd = Twist()
        self.cmd_pub.publish(cmd)

    def status_update(self):
        goal_str = (f"({self.current_goal[0]:.1f}, {self.current_goal[1]:.1f})"
                    if self.current_goal else "none")
        self.get_logger().info(
            f"Coverage: {self.coverage:.0%} | "
            f"Goals reached: {self.goals_reached} | "
            f"Current goal: {goal_str} | "
            f"Done: {self.exploration_done}")


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
