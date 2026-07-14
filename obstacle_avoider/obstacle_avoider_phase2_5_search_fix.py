import math
import random

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan


class obstacle_avoider(Node):

    def __init__(self):
        super().__init__("obstacle_avoider_phase2_5_node")

        self.get_logger().info(
            "Phase 2.5 Squeeze + Corner Obstacle Avoider started."
        )

        self.publisher_ = self.create_publisher(
            TwistStamped,
            "/cmd_vel",
            10,
        )

        self.subscription_ = self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            10,
        )

        # Wall side convention:
        # -1 = left wall
        # +1 = right wall
        #  0 = no selected wall
        self.wall_side = 0
        self.state = "SEARCH"

        # Search/acquisition behavior.
        # +1 = rotate left, -1 = rotate right.
        # Search should rotate instead of blindly driving forward.
        self.search_direction = 1
        self.search_turn_speed = 0.25

        # Avoidance latch:
        # -1 = committed right turn
        # +1 = committed left turn
        #  0 = no committed turn
        self.avoid_direction = 0
        self.avoid_clear_count = 0
        self.avoid_clear_limit = 4

        # Corner recovery memory
        self.wall_lost_count = 0
        self.wall_lost_limit = 5
        self.corner_recovery_count = 0
        self.corner_recovery_limit = 30

        # Squeeze hysteresis
        self.squeeze_clear_count = 0
        self.squeeze_clear_limit = 5

        # Robot geometry / safety margins.
        # Tune these based on your model dimensions.
        self.robot_half_width = 0.18
        self.safety_margin = 0.07
        self.wall_clearance_margin = 0.06

        # Normal wall-following offset. This is the preferred distance,
        # not a hard safety distance.
        self.preferred_wall_distance = 0.65

        # Points closer than this to the fitted wall line are treated as
        # wall points, not obstacles.
        self.wall_ignore_band = 0.08

        # Lookahead region for wall-relative obstacle reasoning.
        self.squeeze_lookahead = 1.10
        self.squeeze_side_limit = 1.40

        # Obstacle thresholds.
        # Hard emergency is intentionally smaller than the old 0.48 m.
        # Larger distances are handled by squeeze/corner logic first.
        self.hard_emergency_distance = 0.22
        self.soft_obstacle_distance = 0.65
        self.collision_clear_distance = 0.75


    def clamp(self, value, low, high):
        return max(low, min(value, high))


    def get_front_distance(self, msg):
        FRONT_HALF_ANGLE = math.radians(30.0)

        front_left = []
        front_right = []

        for i, r in enumerate(msg.ranges):

            if not math.isfinite(r) or r <= 0.0:
                continue

            angle = (
                msg.angle_min
                + i * msg.angle_increment
            )

            # Normalize angle to [-pi, pi]
            angle = math.atan2(
                math.sin(angle),
                math.cos(angle),
            )

            if 0.0 <= angle <= FRONT_HALF_ANGLE:
                front_left.append(r)

            elif -FRONT_HALF_ANGLE <= angle < 0.0:
                front_right.append(r)

        front_ranges = front_left + front_right

        left_clearance = min(
            front_left,
            default=float("inf"),
        )

        right_clearance = min(
            front_right,
            default=float("inf"),
        )

        return (
            front_ranges,
            left_clearance,
            right_clearance,
        )


    def scan_to_points(self, msg):
        points = []

        angle = msg.angle_min

        for r in msg.ranges:

            if math.isfinite(r) and r > 0.0:
                x = r * math.cos(angle)
                y = r * math.sin(angle)

                points.append((x, y))

            angle += msg.angle_increment

        return points


    def filter_side_points(self, points):
        left_points = []
        right_points = []

        for x, y in points:

            if (
                -0.3 < x < 2.0
                and 0.05 < y < 1.2
            ):
                left_points.append((x, y))

            elif (
                -0.3 < x < 2.0
                and -1.2 < y < -0.05
            ):
                right_points.append((x, y))

        return left_points, right_points


    def get_collision_corridor_distance(self, points):
        # Body-aware corridor directly in front of the robot.
        FORWARD_LIMIT = 0.65
        HALF_WIDTH = 0.32

        corridor_distances = []

        for x, y in points:
            if (
                0.0 < x < FORWARD_LIMIT
                and -HALF_WIDTH < y < HALF_WIDTH
            ):
                distance = math.sqrt(x * x + y * y)
                corridor_distances.append(distance)

        return min(
            corridor_distances,
            default=float("inf"),
        )


    def choose_avoid_direction(self, left_clearance, right_clearance):
        # Positive angular z turns left, negative turns right.
        if right_clearance > left_clearance:
            return -1

        if left_clearance > right_clearance:
            return 1

        # If following a wall, turn away from that wall.
        if self.wall_side != 0:
            return -self.wall_side

        return random.choice([-1, 1])


    def reset_avoidance_latch(self):
        self.avoid_direction = 0
        self.avoid_clear_count = 0


    def fit_line(
        self,
        points,
        min_points=20,
        mse_threshold=0.01,
        min_span=0.0,
    ):
        if len(points) < min_points:
            return None

        n = len(points)

        mean_x = (
            sum(x for x, y in points) / n
        )

        mean_y = (
            sum(y for x, y in points) / n
        )

        sxx = sum(
            (x - mean_x) ** 2
            for x, y in points
        )

        syy = sum(
            (y - mean_y) ** 2
            for x, y in points
        )

        sxy = sum(
            (x - mean_x) * (y - mean_y)
            for x, y in points
        )

        angle = 0.5 * math.atan2(
            2 * sxy,
            sxx - syy,
        )

        dx = math.cos(angle)
        dy = math.sin(angle)

        # Unit normal vector to fitted line
        A = -dy
        B = dx

        # Line Ax + By + C = 0 passes through centroid
        C = -(A * mean_x + B * mean_y)

        mse = 0.0
        projections = []

        for x, y in points:
            perpendicular_distance = (
                A * x
                + B * y
                + C
            )

            mse += perpendicular_distance ** 2
            projections.append(x * dx + y * dy)

        mse /= n

        span = max(projections) - min(projections)

        if mse > mse_threshold:
            return None

        if span < min_span:
            return None

        return A, B, C, mse, span, n


    def select_wall(self, left_line, right_line):
        selected_line = None
        selected_side = 0

        # Continue following same wall if still visible.
        if (
            self.wall_side == -1
            and left_line is not None
        ):
            selected_line = left_line
            selected_side = -1

        elif (
            self.wall_side == 1
            and right_line is not None
        ):
            selected_line = right_line
            selected_side = 1

        # No previous wall preference: choose closer valid wall.
        elif (
            left_line is not None
            and right_line is not None
        ):
            left_distance = abs(left_line[2])
            right_distance = abs(right_line[2])

            if left_distance <= right_distance:
                selected_line = left_line
                selected_side = -1

            else:
                selected_line = right_line
                selected_side = 1

        elif left_line is not None:
            selected_line = left_line
            selected_side = -1

        elif right_line is not None:
            selected_line = right_line
            selected_side = 1

        return selected_line, selected_side


    def follow_geometric_wall(
        self,
        cmd,
        line,
        side,
        desired_distance,
        forward_speed=0.15,
    ):
        distance_tolerance = 0.05

        K_angle = 0.35
        K_distance = 0.75

        A, B, C, mse, span, n = line

        # A and B are a unit normal vector.
        # Therefore |C| is distance from robot origin to line.
        actual_distance = abs(C)

        distance_error = (
            desired_distance - actual_distance
        )

        # Direction vector parallel to wall
        wall_dx = B
        wall_dy = -A

        wall_angle = math.atan2(
            wall_dy,
            wall_dx,
        )

        # Select wall direction closest to robot forward (+x)
        if wall_angle > math.pi / 2:
            wall_angle -= math.pi

        elif wall_angle < -math.pi / 2:
            wall_angle += math.pi

        # Ignore very small distance fluctuations
        if abs(distance_error) < distance_tolerance:
            distance_error = 0.0

        away_bias = 0.12

        # Bias away only when too close to the wall.
        if (
            actual_distance
            < desired_distance - distance_tolerance
        ):
            target_angle = (
                wall_angle
                + side * away_bias
            )

        else:
            target_angle = wall_angle

        cmd.twist.linear.x = forward_speed

        angular_z = (
            K_angle * target_angle
            + side
            * K_distance
            * distance_error
        )

        max_turn = 0.35

        angular_z = self.clamp(
            angular_z,
            -max_turn,
            max_turn,
        )

        cmd.twist.angular.z = float(angular_z)

        self.get_logger().info(
            f"Geo follow side={side} | "
            f"desired={desired_distance:.3f}, "
            f"wall_angle={wall_angle:.3f}, "
            f"target_angle={target_angle:.3f}, "
            f"dist={actual_distance:.3f}, "
            f"error={distance_error:.3f}, "
            f"angular={angular_z:.3f}"
        )


    def get_wall_relative_obstacle_info(self, points, line):
        """
        Estimate whether forward obstacle points block the current
        wall-following lane and how large the wall-to-obstacle gap is.

        Distance is measured perpendicular to the fitted wall line.
        """
        A, B, C, mse, span, n = line

        robot_signed_distance = C
        obstacle_distances_from_wall = []

        for x, y in points:
            if not (
                0.0 < x < self.squeeze_lookahead
                and abs(y) < self.squeeze_side_limit
            ):
                continue

            signed_distance = A * x + B * y + C
            distance_from_wall = abs(signed_distance)

            # Ignore points on the opposite side of the wall from the robot.
            if abs(robot_signed_distance) > 1e-6:
                if signed_distance * robot_signed_distance <= 0.0:
                    continue

            # Ignore points that are basically part of the wall itself.
            if distance_from_wall < self.wall_ignore_band:
                continue

            obstacle_distances_from_wall.append(distance_from_wall)

        if not obstacle_distances_from_wall:
            return {
                "has_obstacle": False,
                "gap": None,
                "blocks_current_lane": False,
                "can_squeeze": False,
                "desired_distance": self.preferred_wall_distance,
                "point_count": 0,
            }

        gap = min(obstacle_distances_from_wall)

        lane_margin = self.robot_half_width + self.safety_margin
        lane_inner = max(
            0.0,
            self.preferred_wall_distance - lane_margin,
        )
        lane_outer = self.preferred_wall_distance + lane_margin

        blocks_current_lane = any(
            lane_inner <= d <= lane_outer
            for d in obstacle_distances_from_wall
        )

        minimum_gap = 2.0 * (
            self.robot_half_width
            + self.safety_margin
        )

        minimum_center_distance = (
            self.robot_half_width
            + self.wall_clearance_margin
        )

        can_squeeze = (
            blocks_current_lane
            and gap >= minimum_gap
        )

        desired_distance = self.preferred_wall_distance

        if can_squeeze:
            # Safest path is the middle of the wall-obstacle gap.
            desired_distance = gap / 2.0

            # Never command a centerline too close to the wall.
            desired_distance = max(
                desired_distance,
                minimum_center_distance,
            )

        return {
            "has_obstacle": True,
            "gap": gap,
            "blocks_current_lane": blocks_current_lane,
            "can_squeeze": can_squeeze,
            "desired_distance": desired_distance,
            "point_count": len(obstacle_distances_from_wall),
        }


    def front_blockage_looks_wall_like(self, points):
        front_points = []

        for x, y in points:
            if (
                0.0 < x < 1.0
                and abs(y) < 0.85
            ):
                front_points.append((x, y))

        front_line = self.fit_line(
            front_points,
            min_points=18,
            mse_threshold=0.012,
            min_span=0.45,
        )

        return front_line is not None


    def run_emergency_avoid(self, cmd, left_clearance, right_clearance):
        self.state = "EMERGENCY_AVOID"

        cmd.twist.linear.x = 0.0

        if self.avoid_direction == 0:
            self.avoid_direction = self.choose_avoid_direction(
                left_clearance,
                right_clearance,
            )

        cmd.twist.angular.z = self.avoid_direction * 0.42

        self.get_logger().info(
            f"EMERGENCY_AVOID | direction={self.avoid_direction}"
        )


    def run_inner_corner_turn(self, cmd):
        self.state = "INNER_CORNER_TURN"

        # Turn toward the wall side.
        # left wall (-1)  -> +angular z
        # right wall (+1) -> -angular z
        cmd.twist.linear.x = 0.02
        cmd.twist.angular.z = -self.wall_side * 0.30

        self.get_logger().info(
            f"INNER_CORNER_TURN | wall_side={self.wall_side}, "
            f"angular={cmd.twist.angular.z:.2f}"
        )


    def run_outer_corner_recovery(self, cmd):
        self.state = "OUTER_CORNER_RECOVERY"
        self.corner_recovery_count += 1

        cmd.twist.linear.x = 0.03
        cmd.twist.angular.z = -self.wall_side * 0.25

        self.get_logger().info(
            f"OUTER_CORNER_RECOVERY | wall_side={self.wall_side}, "
            f"lost_count={self.wall_lost_count}, "
            f"recovery_count={self.corner_recovery_count}, "
            f"angular={cmd.twist.angular.z:.2f}"
        )


    def run_search(self, cmd):
        self.state = "SEARCH"

        # In SEARCH, do not drive forward blindly. Rotate in place until
        # a wall enters the left/right ROI and PCA can fit it.
        cmd.twist.linear.x = 0.0
        cmd.twist.angular.z = self.search_direction * self.search_turn_speed

        self.get_logger().info(
            f"SEARCH | rotating to acquire wall, "
            f"direction={self.search_direction}, "
            f"angular={cmd.twist.angular.z:.2f}"
        )


    def scan_callback(self, msg):
        SAFE_DISTANCE = 0.35

        cmd = TwistStamped()

        cmd.header.stamp = (
            self.get_clock().now().to_msg()
        )

        (
            front_ranges,
            left_clearance,
            right_clearance,
        ) = self.get_front_distance(msg)

        front_distance = min(
            front_ranges,
            default=float("inf"),
        )

        points = self.scan_to_points(msg)
        collision_distance = self.get_collision_corridor_distance(points)

        (
            left_points,
            right_points,
        ) = self.filter_side_points(points)

        left_line = self.fit_line(left_points)
        right_line = self.fit_line(right_points)

        selected_line, selected_side = self.select_wall(
            left_line,
            right_line,
        )

        self.get_logger().info(
            f"State={self.state} | "
            f"Front={front_distance:.2f}, "
            f"Collision={collision_distance:.2f}, "
            f"Left={left_clearance:.2f}, "
            f"Right={right_clearance:.2f}, "
            f"wall_side={self.wall_side}, "
            f"avoid_dir={self.avoid_direction}"
        )

        # --------------------------------------------------
        # 1. True hard emergency: immediate safety override.
        # --------------------------------------------------

        if collision_distance < self.hard_emergency_distance:
            self.run_emergency_avoid(
                cmd,
                left_clearance,
                right_clearance,
            )

        # --------------------------------------------------
        # 2. Wall visible: follow, squeeze, or inner-corner turn.
        # --------------------------------------------------

        elif selected_line is not None:
            self.wall_side = selected_side
            self.wall_lost_count = 0
            self.corner_recovery_count = 0

            A, B, C, mse, span, n = selected_line
            wall_name = "Left" if selected_side == -1 else "Right"

            self.get_logger().info(
                f"{wall_name} wall | "
                f"A={A:.3f}, B={B:.3f}, C={C:.3f}, "
                f"mse={mse:.5f}, span={span:.3f}, points={n}"
            )

            obstacle_info = self.get_wall_relative_obstacle_info(
                points,
                selected_line,
            )

            obstacle_ahead = (
                collision_distance < self.soft_obstacle_distance
                or obstacle_info["has_obstacle"]
            )

            if obstacle_ahead:
                gap = obstacle_info["gap"]
                gap_text = "inf" if gap is None else f"{gap:.3f}"

                self.get_logger().info(
                    f"Wall-relative obstacle | "
                    f"gap={gap_text}, "
                    f"blocks_lane={obstacle_info['blocks_current_lane']}, "
                    f"can_squeeze={obstacle_info['can_squeeze']}, "
                    f"desired={obstacle_info['desired_distance']:.3f}, "
                    f"points={obstacle_info['point_count']}"
                )

                if not obstacle_info["blocks_current_lane"]:
                    self.state = "FOLLOW_WALL"
                    self.squeeze_clear_count = 0
                    self.reset_avoidance_latch()

                    self.follow_geometric_wall(
                        cmd,
                        selected_line,
                        selected_side,
                        self.preferred_wall_distance,
                    )

                elif obstacle_info["can_squeeze"]:
                    self.state = "SQUEEZE_THROUGH"
                    self.squeeze_clear_count = 0
                    self.reset_avoidance_latch()

                    self.follow_geometric_wall(
                        cmd,
                        selected_line,
                        selected_side,
                        obstacle_info["desired_distance"],
                        forward_speed=0.10,
                    )

                    self.get_logger().info(
                        "SQUEEZE_THROUGH | using middle of wall-obstacle gap"
                    )

                elif self.front_blockage_looks_wall_like(points):
                    self.run_inner_corner_turn(cmd)

                else:
                    self.run_emergency_avoid(
                        cmd,
                        left_clearance,
                        right_clearance,
                    )

            else:
                self.state = "FOLLOW_WALL"

                if self.state != "SQUEEZE_THROUGH":
                    self.squeeze_clear_count = 0

                self.reset_avoidance_latch()

                self.follow_geometric_wall(
                    cmd,
                    selected_line,
                    selected_side,
                    self.preferred_wall_distance,
                )

        # --------------------------------------------------
        # 3. Wall lost while previously following: outer corner.
        # --------------------------------------------------

        elif self.wall_side != 0:
            self.wall_lost_count += 1

            # If there is no wall and front is blocked, obstacle avoidance is
            # safer than assuming an outer corner.
            if collision_distance < self.soft_obstacle_distance:
                self.run_emergency_avoid(
                    cmd,
                    left_clearance,
                    right_clearance,
                )

            elif self.corner_recovery_count < self.corner_recovery_limit:
                self.run_outer_corner_recovery(cmd)

            else:
                self.get_logger().info(
                    "OUTER_CORNER_TIMEOUT | resetting wall memory"
                )
                previous_wall_side = self.wall_side
                if previous_wall_side != 0:
                    self.search_direction = -previous_wall_side

                self.wall_side = 0
                self.wall_lost_count = 0
                self.corner_recovery_count = 0
                self.reset_avoidance_latch()

                self.run_search(cmd)

        # --------------------------------------------------
        # 4. No wall memory and obstacle ahead.
        # --------------------------------------------------

        elif collision_distance < self.soft_obstacle_distance:
            self.run_emergency_avoid(
                cmd,
                left_clearance,
                right_clearance,
            )

        elif front_distance < SAFE_DISTANCE:
            self.run_emergency_avoid(
                cmd,
                left_clearance,
                right_clearance,
            )

        # --------------------------------------------------
        # 5. Search.
        # --------------------------------------------------

        else:
            self.wall_side = 0
            self.wall_lost_count = 0
            self.corner_recovery_count = 0
            self.reset_avoidance_latch()

            self.run_search(cmd)

        self.publisher_.publish(cmd)


def main(args=None):
    rclpy.init(args=args)

    node = obstacle_avoider()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
