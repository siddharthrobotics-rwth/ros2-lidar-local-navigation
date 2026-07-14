import math
import random

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan


class obstacle_avoider(Node):

    def __init__(self):
        super().__init__("obstacle_avoider_node")

        self.get_logger().info(
            "Obstacle Avoider Node has been started."
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

        self.turn_direction = 0
        self.wall_side = 0

        # Obstacle avoidance latch:
        # -1 = committed right turn
        # +1 = committed left turn
        #  0 = no committed avoidance direction
        self.avoid_direction = 0
        self.avoid_clear_count = 0
        self.avoid_clear_limit = 4


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
        # This catches obstacles that are not exactly in the narrow center ray
        # but are still inside the robot's physical driving path.
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

        # If already following a wall, turn away from that wall.
        if self.wall_side != 0:
            return -self.wall_side

        return random.choice([-1, 1])


    def reset_avoidance_latch(self):
        self.avoid_direction = 0
        self.avoid_clear_count = 0


    def fit_line(self, points):
        MIN_WALL_POINTS = 20
        MSE_THRESHOLD = 0.01

        if len(points) < MIN_WALL_POINTS:
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

        for x, y in points:
            perpendicular_distance = (
                A * x
                + B * y
                + C
            )

            mse += perpendicular_distance ** 2

        mse /= n

        if mse > MSE_THRESHOLD:
            return None

        return A, B, C, mse


    def follow_geometric_wall(self, cmd, line, side):
        desired_distance = 0.65
        distance_tolerance = 0.05

        K_angle = 0.35
        K_distance = 0.75

        A, B, C, mse = line

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

        # Bias away only when too close
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

        cmd.twist.linear.x = 0.15

        # Corrected steering signs
        angular_z = (
            K_angle * target_angle
            + side
            * K_distance
            * distance_error
        )

        max_turn = 0.35

        angular_z = max(
            min(angular_z, max_turn),
            -max_turn,
        )

        cmd.twist.angular.z = float(angular_z)

        self.get_logger().info(
            f"Geo follow side={side} | "
            f"wall_angle={wall_angle:.3f}, "
            f"target_angle={target_angle:.3f}, "
            f"dist={actual_distance:.3f}, "
            f"error={distance_error:.3f}, "
            f"angular={angular_z:.3f}"
        )


    def scan_callback(self, msg):
        SAFE_DISTANCE = 0.35
        CRITICAL_DISTANCE = 0.20

        # Robust obstacle interpretation thresholds
        COLLISION_CORRIDOR_DISTANCE = 0.48
        COLLISION_CLEAR_DISTANCE = 0.70

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

        self.get_logger().info(
            f"Front={front_distance:.2f}, "
            f"Collision={collision_distance:.2f}, "
            f"Left={left_clearance:.2f}, "
            f"Right={right_clearance:.2f}, "
            f"avoid_dir={self.avoid_direction}, "
            f"avoid_clear={self.avoid_clear_count}"
        )

        # --------------------------------------------------
        # 1. Robust obstacle override
        # --------------------------------------------------

        obstacle_too_close = (
            collision_distance < COLLISION_CORRIDOR_DISTANCE
            or front_distance < CRITICAL_DISTANCE
        )

        still_clearing_obstacle = (
            self.avoid_direction != 0
            and collision_distance < COLLISION_CLEAR_DISTANCE
        )

        if obstacle_too_close or still_clearing_obstacle:

            cmd.twist.linear.x = 0.0

            # Choose once and keep the same turn direction.
            # This prevents left/right flipping every scan.
            if self.avoid_direction == 0:
                self.avoid_direction = self.choose_avoid_direction(
                    left_clearance,
                    right_clearance,
                )

            if obstacle_too_close:
                cmd.twist.angular.z = self.avoid_direction * 0.42
                self.avoid_clear_count = 0

                self.get_logger().info(
                    f"OBSTACLE_OVERRIDE | collision={collision_distance:.2f}, "
                    f"front={front_distance:.2f}, "
                    f"direction={self.avoid_direction}"
                )

            else:
                cmd.twist.angular.z = self.avoid_direction * 0.35
                self.avoid_clear_count += 1

                self.get_logger().info(
                    f"OBSTACLE_CLEARING | continuing direction={self.avoid_direction}, "
                    f"collision={collision_distance:.2f}, "
                    f"clear_count={self.avoid_clear_count}"
                )

        # --------------------------------------------------
        # 2. Select geometric wall
        # --------------------------------------------------

        else:
            if self.avoid_direction != 0:
                self.avoid_clear_count += 1

                if self.avoid_clear_count >= self.avoid_clear_limit:
                    self.get_logger().info(
                        "OBSTACLE_CLEAR | releasing avoidance latch"
                    )
                    self.reset_avoidance_latch()

            selected_line = None
            selected_side = 0

            # Continue following same wall if still visible
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

            # No previous wall preference:
            # choose closer valid wall
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

            # --------------------------------------------------
            # 3. Geometric wall following
            # --------------------------------------------------

            if selected_line is not None:

                self.wall_side = selected_side

                A, B, C, mse = selected_line

                wall_name = (
                    "Left"
                    if selected_side == -1
                    else "Right"
                )

                self.get_logger().info(
                    f"{wall_name} wall | "
                    f"A={A:.3f}, "
                    f"B={B:.3f}, "
                    f"C={C:.3f}, "
                    f"mse={mse:.5f}"
                )

                self.follow_geometric_wall(
                    cmd,
                    selected_line,
                    selected_side,
                )

            # --------------------------------------------------
            # 4. Front obstacle, but no valid wall
            # --------------------------------------------------

            elif front_distance < SAFE_DISTANCE:

                self.wall_side = 0
                cmd.twist.linear.x = 0.0

                if self.avoid_direction == 0:
                    self.avoid_direction = self.choose_avoid_direction(
                        left_clearance,
                        right_clearance,
                    )

                cmd.twist.angular.z = self.avoid_direction * 0.40

                self.get_logger().info(
                    f"Front obstacle, no valid wall -> avoiding, "
                    f"direction={self.avoid_direction}"
                )

            # --------------------------------------------------
            # 5. Search
            # --------------------------------------------------

            else:

                self.wall_side = 0

                cmd.twist.linear.x = 0.12
                cmd.twist.angular.z = 0.0

                self.get_logger().info(
                    "No valid wall detected -> searching"
                )

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