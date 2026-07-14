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

        # --------------------------------------------------
        # Navigation state
        # --------------------------------------------------

        self.state = "SEARCH_FOR_WALL"

        # Wall side convention:
        # -1 = wall on robot left
        # +1 = wall on robot right
        #  0 = no selected wall
        self.wall_side = 0

        # --------------------------------------------------
        # Target wall lock
        # --------------------------------------------------
        # The robot should not abandon a selected boundary wall just
        # because a closer obstacle/cylinder appears.
        # It only unlocks after a proper missing timeout.

        self.target_wall_locked = False
        self.target_wall = None
        self.target_wall_side = 0
        self.target_wall_missing_count = 0
        self.target_wall_missing_limit = 45

        # Used for short-term consistency scoring
        self.last_target_angle = None
        self.last_target_distance = None

        # --------------------------------------------------
        # Close boundary-following parameters
        # --------------------------------------------------

        # Distance is from robot/LiDAR frame to fitted wall line.
        # Keep this small enough to hug the boundary, but not so small
        # that the robot body scrapes/stutters against the wall.
        self.desired_wall_distance = 0.30
        self.distance_tolerance = 0.06
        self.approach_margin = 0.12

        # --------------------------------------------------
        # Recovery / open-space behavior
        # --------------------------------------------------

        self.open_space_distance = 4.0
        self.recovery_count = 0
        self.recovery_limit = 45

        # --------------------------------------------------
        # Obstacle avoidance latch
        # --------------------------------------------------
        # -1 = committed right turn
        # +1 = committed left turn
        #  0 = no committed avoidance direction

        self.avoid_direction = 0
        self.avoid_clear_count = 0
        self.avoid_clear_limit = 5

        # --------------------------------------------------
        # 360-degree wall candidate extraction parameters
        # --------------------------------------------------

        self.cluster_jump_distance = 0.22
        self.min_cluster_points = 10
        self.min_wall_points = 14
        self.min_line_span = 0.55
        self.max_line_mse = 0.012
        self.max_candidate_distance = 2.50

        # Ignore candidates whose centroid is very close to the robot body.
        # These are usually obstacles, not useful boundaries to follow.
        self.min_candidate_range = 0.25


    # ==================================================
    # Basic scan helpers
    # ==================================================

    def normalize_angle(self, angle):
        return math.atan2(
            math.sin(angle),
            math.cos(angle),
        )


    def angle_difference(self, a, b):
        return abs(
            self.normalize_angle(a - b)
        )


    def get_front_distance(self, msg):
        FRONT_HALF_ANGLE = math.radians(25.0)

        front_left = []
        front_right = []

        for i, r in enumerate(msg.ranges):

            if not math.isfinite(r) or r <= 0.0:
                continue

            angle = (
                msg.angle_min
                + i * msg.angle_increment
            )

            angle = self.normalize_angle(angle)

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

                points.append(
                    {
                        "x": x,
                        "y": y,
                        "r": r,
                        "angle": self.normalize_angle(angle),
                    }
                )

            angle += msg.angle_increment

        return points


    def get_collision_corridor_distance(self, points):
        # Rectangular safety corridor directly in front of the robot body.
        # This remains independent of wall target selection.
        FORWARD_LIMIT = 0.55
        HALF_WIDTH = 0.30

        corridor_distances = []

        for p in points:
            x = p["x"]
            y = p["y"]

            if (
                0.0 < x < FORWARD_LIMIT
                and -HALF_WIDTH < y < HALF_WIDTH
            ):
                corridor_distances.append(
                    math.sqrt(x * x + y * y)
                )

        return min(
            corridor_distances,
            default=float("inf"),
        )


    # ==================================================
    # 360-degree clustering and line extraction
    # ==================================================

    def cluster_scan_points(self, points):
        # Clustering does not decide "wall".
        # It only separates continuous scan surfaces.
        # Wall-likeness is checked later using PCA span/MSE scoring.

        clusters = []
        current = []

        previous = None

        for p in points:

            if previous is None:
                current = [p]
                previous = p
                continue

            dx = p["x"] - previous["x"]
            dy = p["y"] - previous["y"]

            gap = math.sqrt(dx * dx + dy * dy)

            adaptive_gap = max(
                self.cluster_jump_distance,
                0.12 * min(p["r"], previous["r"]),
            )

            if gap <= adaptive_gap:
                current.append(p)

            else:
                if len(current) >= self.min_cluster_points:
                    clusters.append(current)

                current = [p]

            previous = p

        if len(current) >= self.min_cluster_points:
            clusters.append(current)

        return clusters


    def fit_line_to_cluster(self, cluster):
        if len(cluster) < self.min_wall_points:
            return None

        points = [
            (p["x"], p["y"])
            for p in cluster
        ]

        n = len(points)

        mean_x = (
            sum(x for x, y in points) / n
        )

        mean_y = (
            sum(y for x, y in points) / n
        )

        centroid_range = math.sqrt(
            mean_x * mean_x
            + mean_y * mean_y
        )

        if centroid_range < self.min_candidate_range:
            return None

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

        # Unit normal to the fitted line
        A = -dy
        B = dx

        C = -(
            A * mean_x
            + B * mean_y
        )

        mse = 0.0

        projections = []

        for x, y in points:
            perpendicular_distance = (
                A * x
                + B * y
                + C
            )

            mse += perpendicular_distance ** 2

            projection = (
                x * dx
                + y * dy
            )

            projections.append(projection)

        mse /= n

        span = (
            max(projections)
            - min(projections)
        )

        if mse > self.max_line_mse:
            return None

        if span < self.min_line_span:
            return None

        distance = abs(C)

        if distance > self.max_candidate_distance:
            return None

        # Wall direction closest to robot forward (+x)
        wall_angle = math.atan2(dy, dx)

        if wall_angle > math.pi / 2:
            wall_angle -= math.pi

        elif wall_angle < -math.pi / 2:
            wall_angle += math.pi

        # Side is based on where the candidate mostly lies in robot frame.
        # This preserves the old left/right convention but no longer uses
        # hard rectangular ROIs to detect walls.
        if mean_y >= 0.0:
            side = -1

        else:
            side = 1

        return {
            "A": A,
            "B": B,
            "C": C,
            "mse": mse,
            "span": span,
            "n": n,
            "mean_x": mean_x,
            "mean_y": mean_y,
            "distance": distance,
            "wall_angle": wall_angle,
            "side": side,
        }


    def extract_wall_candidates(self, points):
        clusters = self.cluster_scan_points(points)

        candidates = []

        for cluster in clusters:
            candidate = self.fit_line_to_cluster(cluster)

            if candidate is not None:
                candidates.append(candidate)

        return candidates


    # ==================================================
    # Candidate scoring and target-wall lock
    # ==================================================

    def clamp01(self, value):
        return max(
            0.0,
            min(1.0, value),
        )


    def wall_candidate_score(self, candidate, prefer_locked_target=False):
        # Normalize terms to roughly [0, 1].
        # The goal is not mathematical perfection; it is stable behavior.

        span_score = self.clamp01(
            candidate["span"] / 1.50
        )

        mse_score = self.clamp01(
            1.0 - candidate["mse"] / self.max_line_mse
        )

        distance = candidate["distance"]

        # Prefer walls that are reachable and not extremely far.
        # Distance should not dominate the score, because close cylinders
        # are dangerous false positives.
        if distance < self.desired_wall_distance:
            distance_score = self.clamp01(
                distance / self.desired_wall_distance
            )

        else:
            distance_score = self.clamp01(
                1.0 - (
                    distance - self.desired_wall_distance
                ) / 2.0
            )

        point_score = self.clamp01(
            candidate["n"] / 60.0
        )

        side_score = 1.0

        # Discourage candidates directly behind the robot unless we are
        # explicitly recovering. Boundary following should be forward-usable.
        forward_score = self.clamp01(
            (candidate["mean_x"] + 0.30) / 1.30
        )

        persistence_score = 0.0

        if prefer_locked_target:
            if candidate["side"] == self.target_wall_side:
                persistence_score += 0.45

            if self.last_target_angle is not None:
                angle_error = self.angle_difference(
                    candidate["wall_angle"],
                    self.last_target_angle,
                )

                persistence_score += 0.35 * self.clamp01(
                    1.0 - angle_error / math.radians(35.0)
                )

            if self.last_target_distance is not None:
                distance_error = abs(
                    candidate["distance"]
                    - self.last_target_distance
                )

                persistence_score += 0.20 * self.clamp01(
                    1.0 - distance_error / 0.55
                )

        score = (
            2.20 * span_score
            + 1.80 * mse_score
            + 0.70 * distance_score
            + 0.50 * point_score
            + 0.50 * forward_score
            + 2.50 * persistence_score
            + 0.30 * side_score
        )

        return score


    def choose_best_wall_candidate(self, candidates):
        if not candidates:
            return None

        best_candidate = None
        best_score = -float("inf")

        for candidate in candidates:
            score = self.wall_candidate_score(
                candidate,
                prefer_locked_target=False,
            )

            candidate["score"] = score

            if score > best_score:
                best_score = score
                best_candidate = candidate

        return best_candidate


    def find_matching_locked_wall(self, candidates):
        if (
            not self.target_wall_locked
            or self.target_wall_side == 0
        ):
            return None

        matching_candidates = []

        for candidate in candidates:

            # Hard rule: once locked, do not jump to the opposite side
            # unless the target has fully timed out and unlocked.
            if candidate["side"] != self.target_wall_side:
                continue

            # Soft consistency rules.
            angle_ok = True
            distance_ok = True

            if self.last_target_angle is not None:
                angle_error = self.angle_difference(
                    candidate["wall_angle"],
                    self.last_target_angle,
                )

                angle_ok = angle_error < math.radians(45.0)

            if self.last_target_distance is not None:
                distance_error = abs(
                    candidate["distance"]
                    - self.last_target_distance
                )

                # Allow distance to change while approaching the wall,
                # but prevent jumping to a totally different nearby object.
                distance_ok = distance_error < 0.90

            if angle_ok and distance_ok:
                matching_candidates.append(candidate)

        if not matching_candidates:
            return None

        best_candidate = None
        best_score = -float("inf")

        for candidate in matching_candidates:
            score = self.wall_candidate_score(
                candidate,
                prefer_locked_target=True,
            )

            candidate["score"] = score

            if score > best_score:
                best_score = score
                best_candidate = candidate

        return best_candidate


    def lock_target_wall(self, candidate):
        self.target_wall_locked = True
        self.target_wall = candidate
        self.target_wall_side = candidate["side"]
        self.wall_side = candidate["side"]

        self.target_wall_missing_count = 0
        self.recovery_count = 0

        self.last_target_angle = candidate["wall_angle"]
        self.last_target_distance = candidate["distance"]

        self.get_logger().info(
            f"TARGET_WALL_LOCKED | side={self.target_wall_side}, "
            f"dist={candidate['distance']:.3f}, "
            f"angle={candidate['wall_angle']:.3f}, "
            f"span={candidate['span']:.3f}, "
            f"mse={candidate['mse']:.5f}, "
            f"points={candidate['n']}, "
            f"score={candidate.get('score', 0.0):.3f}"
        )


    def update_target_wall(self, candidate):
        self.target_wall = candidate
        self.target_wall_side = candidate["side"]
        self.wall_side = candidate["side"]

        self.target_wall_missing_count = 0
        self.recovery_count = 0

        self.last_target_angle = candidate["wall_angle"]
        self.last_target_distance = candidate["distance"]


    def unlock_target_wall(self, reason):
        self.get_logger().info(
            f"TARGET_WALL_UNLOCKED | reason={reason}"
        )

        self.target_wall_locked = False
        self.target_wall = None
        self.target_wall_side = 0
        self.wall_side = 0
        self.target_wall_missing_count = 0
        self.recovery_count = 0
        self.last_target_angle = None
        self.last_target_distance = None


    # ==================================================
    # Motion control
    # ==================================================

    def candidate_to_line_tuple(self, candidate):
        return (
            candidate["A"],
            candidate["B"],
            candidate["C"],
            candidate["mse"],
        )


    def follow_or_approach_target_wall(self, cmd, candidate):
        desired_distance = self.desired_wall_distance
        distance_tolerance = self.distance_tolerance

        K_angle = 0.35
        K_distance = 0.45

        side = candidate["side"]

        actual_distance = candidate["distance"]
        distance_error = (
            desired_distance
            - actual_distance
        )

        wall_angle = candidate["wall_angle"]

        if abs(distance_error) < distance_tolerance:
            distance_error = 0.0

        # Approach/follow state split:
        # far from wall -> explicitly close the distance
        # near wall     -> follow parallel at desired offset
        if actual_distance > desired_distance + self.approach_margin:
            self.state = "APPROACH_TARGET_WALL"
            cmd.twist.linear.x = 0.08

        else:
            self.state = "FOLLOW_TARGET_WALL"
            cmd.twist.linear.x = 0.10

        away_bias = 0.10

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
            f"{self.state} | side={side}, "
            f"dist={actual_distance:.3f}, "
            f"desired={desired_distance:.3f}, "
            f"error={distance_error:.3f}, "
            f"angle={wall_angle:.3f}, "
            f"target_angle={target_angle:.3f}, "
            f"span={candidate['span']:.3f}, "
            f"mse={candidate['mse']:.5f}, "
            f"score={candidate.get('score', 0.0):.3f}, "
            f"angular={angular_z:.3f}"
        )


    def run_target_recovery(self, cmd):
        self.state = "REACQUIRE_TARGET_WALL"

        cmd.twist.linear.x = 0.0

        if self.target_wall_side == -1:
            cmd.twist.angular.z = 0.28

        elif self.target_wall_side == 1:
            cmd.twist.angular.z = -0.28

        else:
            cmd.twist.angular.z = 0.25

        self.get_logger().info(
            f"REACQUIRE_TARGET_WALL | locked_side={self.target_wall_side}, "
            f"missing_count={self.target_wall_missing_count}, "
            f"recovery_count={self.recovery_count}"
        )


    def run_search_for_wall(self, cmd):
        self.state = "SEARCH_FOR_WALL"

        cmd.twist.linear.x = 0.06
        cmd.twist.angular.z = 0.22

        self.get_logger().info(
            "SEARCH_FOR_WALL | scanning 360 for robust wall candidate"
        )


    def choose_avoid_direction(self, left_clearance, right_clearance):
        if right_clearance > left_clearance:
            return -1

        if left_clearance > right_clearance:
            return 1

        # While a wall target is locked, avoid away from the target side
        # but never delete the target wall memory.
        if self.target_wall_locked and self.target_wall_side != 0:
            return -self.target_wall_side

        if self.wall_side != 0:
            return -self.wall_side

        return random.choice([-1, 1])


    def reset_avoidance_latch(self):
        self.avoid_direction = 0
        self.avoid_clear_count = 0


    # ==================================================
    # Main callback
    # ==================================================

    def scan_callback(self, msg):
        SAFE_DISTANCE = 0.35
        CRITICAL_DISTANCE = 0.15

        COLLISION_CORRIDOR_DISTANCE = 0.42
        COLLISION_CLEAR_DISTANCE = 0.65

        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()

        front_ranges, left_clearance, right_clearance = self.get_front_distance(msg)

        front_distance = min(
            front_ranges,
            default=float("inf"),
        )

        points = self.scan_to_points(msg)
        collision_distance = self.get_collision_corridor_distance(points)

        candidates = self.extract_wall_candidates(points)

        locked_candidate = self.find_matching_locked_wall(candidates)

        self.get_logger().info(
            f"State={self.state} | Front={front_distance:.2f}, "
            f"Collision={collision_distance:.2f}, "
            f"Left={left_clearance:.2f}, Right={right_clearance:.2f}, "
            f"candidates={len(candidates)}, "
            f"locked={self.target_wall_locked}, "
            f"target_side={self.target_wall_side}, "
            f"missing={self.target_wall_missing_count}, "
            f"avoid_dir={self.avoid_direction}"
        )

        # --------------------------------------------------
        # 1. Safety override: avoid obstacle without forgetting target wall
        # --------------------------------------------------

        if collision_distance < COLLISION_CORRIDOR_DISTANCE:
            self.state = "AVOID_OBSTACLE_KEEP_TARGET"

            cmd.twist.linear.x = 0.0

            if self.avoid_direction == 0:
                self.avoid_direction = self.choose_avoid_direction(
                    left_clearance,
                    right_clearance,
                )

            cmd.twist.angular.z = self.avoid_direction * 0.40
            self.avoid_clear_count = 0

            self.get_logger().info(
                f"AVOID_OBSTACLE_KEEP_TARGET | collision={collision_distance:.2f}, "
                f"direction={self.avoid_direction}, "
                f"target_locked={self.target_wall_locked}"
            )

        elif (
            self.avoid_direction != 0
            and collision_distance < COLLISION_CLEAR_DISTANCE
        ):
            self.state = "CLEARING_OBSTACLE_KEEP_TARGET"

            cmd.twist.linear.x = 0.0
            cmd.twist.angular.z = self.avoid_direction * 0.35

            self.avoid_clear_count += 1

            self.get_logger().info(
                f"CLEARING_OBSTACLE_KEEP_TARGET | collision={collision_distance:.2f}, "
                f"direction={self.avoid_direction}, "
                f"clear_count={self.avoid_clear_count}, "
                f"target_locked={self.target_wall_locked}"
            )

        elif front_distance < CRITICAL_DISTANCE:
            self.state = "EMERGENCY_AVOID_KEEP_TARGET"

            cmd.twist.linear.x = 0.0

            if self.avoid_direction == 0:
                self.avoid_direction = self.choose_avoid_direction(
                    left_clearance,
                    right_clearance,
                )

            cmd.twist.angular.z = self.avoid_direction * 0.35

            self.get_logger().info(
                f"EMERGENCY_AVOID_KEEP_TARGET | front={front_distance:.2f}, "
                f"direction={self.avoid_direction}, "
                f"target_locked={self.target_wall_locked}"
            )

        # --------------------------------------------------
        # 2. If a target wall is already locked, only follow/reacquire that wall
        # --------------------------------------------------

        elif self.target_wall_locked:
            self.reset_avoidance_latch()

            if locked_candidate is not None:
                self.update_target_wall(locked_candidate)
                self.follow_or_approach_target_wall(
                    cmd,
                    locked_candidate,
                )

            else:
                self.target_wall_missing_count += 1
                self.recovery_count += 1

                if (
                    self.target_wall_missing_count
                    > self.target_wall_missing_limit
                    or self.recovery_count
                    > self.recovery_limit
                ):
                    self.unlock_target_wall(
                        "target missing timeout"
                    )
                    self.run_search_for_wall(cmd)

                else:
                    self.run_target_recovery(cmd)

        # --------------------------------------------------
        # 3. No target locked: choose robust wall candidate using weighted score
        # --------------------------------------------------

        else:
            self.reset_avoidance_latch()

            best_candidate = self.choose_best_wall_candidate(candidates)

            if best_candidate is not None:
                self.lock_target_wall(best_candidate)
                self.follow_or_approach_target_wall(
                    cmd,
                    best_candidate,
                )

            elif front_distance < SAFE_DISTANCE:
                self.state = "AVOID_OBSTACLE_NO_TARGET"

                cmd.twist.linear.x = 0.0

                if self.avoid_direction == 0:
                    self.avoid_direction = self.choose_avoid_direction(
                        left_clearance,
                        right_clearance,
                    )

                cmd.twist.angular.z = self.avoid_direction * 0.35

                self.get_logger().info(
                    f"AVOID_OBSTACLE_NO_TARGET | direction={self.avoid_direction}"
                )

            else:
                self.run_search_for_wall(cmd)

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