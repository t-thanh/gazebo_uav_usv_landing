# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPS-denied autonomous landing of an MRS X500 PX4 quadrotor UAV onto a moving Otter USV using only onboard sensors (gimbal camera, Garmin rangefinder, IMU). Control via ROS/MAVROS. Simulation in Gazebo (ROS Noetic, Ubuntu 20.04).

## Repository Layout

```
ros/                          # Catkin workspace root
  src/
    usv_sim/                  # git submodule: fork of jhlenes/usv_simulator
    uav_sim/
      simulation/             # git submodule: t-thanh/simulation (MRS+PX4, gitman-managed)
      uav_core/               # git submodule: ctu-mrs/uav_core (gitman-managed)
    ros_landing/              # all custom landing packages (this repo)
      ar_code_landing/        # ArUco/AprilTag landing pipeline (no GPS, pure MAVROS)
      uav_traj_landing/       # Trajectory planning + Sliding Mode Controller landing
      uav_landing_demo/       # Top-level integration: world + spawners + demo launch
      uav_landing_evaluator/  # Drop-test evaluation framework (CSV output)
      uav_landing_gui/        # Qt5 GUI (telemetry, mode selection, evaluator results)
      ocean_rangefinder/      # Virtual Garmin rangefinder filter for ocean surface
docker/                       # Dockerfile for reproducible ROS Noetic environment
paper/                        # Figures and data for the paper
reports/                      # Evaluation reports
tmp/                          # Scratch / temporary development (gitignored)
```

**Separate repos (added as submodules later):**
- `otter_usv_detector` — YOLO OBB detection of the Otter from aerial view + custom msgs
- `uav_gimbal` — Gremsy G-Hadron 3-axis gimbal simulation (URDF, ros_control)
- `gimbal_usv_tracker` — IBVS gimbal servo controller

## Workspace Setup

```bash
# After cloning, init submodules
git submodule update --init --recursive

# Install gitman for uav_sim dependencies
pip install gitman
cd ros/src/uav_sim/simulation && gitman install
cd ros/src/uav_sim/uav_core && gitman install

# Build
cd ros
source /opt/ros/noetic/setup.bash
catkin build

source ros/devel/setup.bash
```

## Key Packages

### ar_code_landing
Landing pipeline using nested ArUco/AprilTag markers (outer: DICT_APRILTAG_36h11 ID=10, inner: DICT_4X4_50 ID=1) on the USV deck.
- `gimbal_aruco_tracker_node.py` — keeps gimbal centred on the ArUco marker
- `ar_landing_controller_node.py` — attitude + thrust control at 50 Hz via `mavros/setpoint_raw/attitude`
- `ar_landing_sm_node.py` — state machine: `PRE_ARMED → ARMING → CLIMBING → ALIGN → DESCEND → LANDED`

### uav_traj_landing
Replaces the PID controller in ar_code_landing with:
1. Minimum-jerk polynomial trajectory generator
2. Sliding Mode Controller with Super-Twisting Algorithm (from `scripts/controllers/smc.py`)

Reuses `gimbal_aruco_tracker_node.py` from `ar_code_landing`.

### uav_landing_demo
Integration package that spawns the full scene: Otter USV world + MRS drone + gimbal. Entry point for the full demo.

### ocean_rangefinder
Filters raw Garmin LiDAR Lite rangefinder readings to remove ocean wave returns, publishing a clean altitude above the USV deck.

### uav_landing_evaluator
Drop-test framework: teleports a free-falling rigid body above the USV, records contact energy/angle/position/fall time, saves CSV.

## Common Launch Sequence

```bash
# Full simulation with ArUco landing
roslaunch uav_landing_demo moving_usv_env.launch
roslaunch ar_code_landing usv_uav_gimbal_landing.launch

# Trajectory-based landing
roslaunch uav_traj_landing traj_landing.launch

# Drop-test evaluation
roslaunch uav_landing_evaluator landing_evaluator.launch
```

## usv_sim Modifications (vs upstream jhlenes/usv_simulator)

The Otter model has been modified to carry a nested ArUco landing target:
- `otter_description/urdf/otter_base.urdf.xacro` — added `ar_code` link with Target_2 mesh (scale 2.5×2.5×0.05, mounted 0.7 m above base, rotated 90° so bow appears at top when viewed from above)
- `otter_description/meshes/otter/otter.dae` — modified mesh
- `otter_gazebo/urdf/Target_2/` — mesh assets for the nested ArUco board (6 .dae variants)
- New launch files: `usv_uav.launch`, `usv_uav_gimbal.launch`, `start_uav.launch`
- New world files: `usv_uav.world.xacro`, `usv_uav_fast.world.xacro`

## uav_sim Stack

Uses **gitman** (not git submodules) for large external dependencies. Run `gitman install` inside `simulation/` and `uav_core/` after cloning.
- `simulation/` — forked from `ctu-mrs/simulation` at commit `8824eb7`, adapted for gimbal integration with X500
- `uav_core/` — from `ctu-mrs/uav_core`, unmodified; provides mavros, mrs_lib, mrs_msgs, mrs_uav_controllers
