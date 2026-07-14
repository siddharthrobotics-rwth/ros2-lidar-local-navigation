# ROS 2 LiDAR-Based Local Navigation System

A custom local navigation controller for a differential-drive robot using **ROS 2 Jazzy, Python, Gazebo, and 2D LiDAR**.

The project is built from scratch without relying on Nav2. It processes `LaserScan` data, converts it into a local Cartesian point cloud, estimates wall geometry using **PCA / total least squares**, and controls the robot using geometric feedback and finite-state behavior logic.

## Features

- 2D LiDAR `LaserScan` processing
- Polar-to-Cartesian local point-cloud conversion
- Left and right wall candidate extraction
- PCA / total least-squares wall-line fitting
- MSE-based line validation
- Geometric wall-following controller
- Wall-angle and perpendicular-distance regulation
- Body-width collision-corridor detection
- Latched obstacle avoidance
- Wall-side memory
- Wall search and reacquisition
- Inner and outer corner handling
- Wall-relative obstacle reasoning
- Dynamic squeeze-through behavior based on available clearance
- Finite-state navigation architecture

## Navigation States

- `SEARCH`
- `FOLLOW_WALL`
- `SQUEEZE_THROUGH`
- `INNER_CORNER_TURN`
- `OUTER_CORNER_RECOVERY`
- `EMERGENCY_AVOID`

## Architecture

### Perception

The node subscribes to `/scan`.

LiDAR ranges are converted into Cartesian points in the robot frame, where `x` points forward and `y` points left.

Wall candidates are extracted from local regions and fitted using PCA / total least squares.

Wall lines are represented as `Ax + By + C = 0`.

Because `(A, B)` is a unit normal vector, the perpendicular robot-to-wall distance is `|C|`.

### Control

The geometric wall-following controller combines:

- wall orientation error
- perpendicular distance error
- angular velocity limiting
- forward velocity control

The robot publishes `TwistStamped` commands to `/cmd_vel`.

### Behavior Layer

A finite-state architecture coordinates wall following, obstacle handling, corner recovery, squeeze-through behavior, and wall search.

## Package Structure

```text
obstacle_avoider/
├── obstacle_avoider/
│   ├── obstacle_avoider.py
│   ├── obstacle_avoider_phase2_robust_obstacle.py
│   ├── obstacle_avoider_phase2_5_squeeze_corner.py
│   ├── obstacle_avoider_phase2_5_search_fix.py
│   └── obstacle_avoider_phase3.py
├── resource/
├── test/
├── package.xml
├── setup.cfg
└── setup.py
```

## Build

```bash
cd ~/ros2_ws
colcon build --packages-select obstacle_avoider --symlink-install
source /opt/ros/jazzy/setup.bash
source install/setup.bash
```

## Run

### Basic wall-following node

```bash
ros2 run obstacle_avoider obstacle_avoider_node
```

### Robust obstacle-avoidance version

```bash
ros2 run obstacle_avoider obstacle_avoider_phase2_robust
```

### Squeeze-through and corner-handling version

```bash
ros2 run obstacle_avoider obstacle_avoider_phase2_5
```

### Search and wall-reacquisition version

```bash
ros2 run obstacle_avoider obstacle_avoider_phase2_5_search_fix
```

## Technologies

- ROS 2 Jazzy
- Python
- Gazebo
- 2D LiDAR
- `sensor_msgs/LaserScan`
- `geometry_msgs/TwistStamped`
- PCA
- Total least squares
- 2D geometry
- Geometric feedback control
- Finite-state machines

## Current Status

The project is under active development.

Current work focuses on:

- reducing state-transition oscillations
- improving obstacle interpretation in cluttered environments
- making squeeze-through behavior more robust
- improving wall acquisition and reacquisition
- adding clearer visualization and debugging tools

## Scope

This project is a custom local navigation and wall-following system. It does not currently implement SLAM, global path planning, localization, or full Nav2 functionality.
